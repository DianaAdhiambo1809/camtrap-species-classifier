# =====================================================================
# Train a CNN, with hyperparameters chosen by Optuna
# =====================================================================
# Two models can be trained:
#   scratch     a small CNN built layer by layer (the Lab 7 design, one block deeper for 224px photos)
#   pretrained  EfficientNetB0 trained on ImageNet, with a new final layer for our 14 classes
#
# Optuna searches the settings that give the best macro-F1 on the VALIDATION cameras.
# The test cameras are not used here; they are kept for the final evaluation.
#
# Needs TensorFlow, Optuna, MLflow and the photos in data/images/, so run it in Colab:
#   python -m camtrap.models.train --model scratch
#   python -m camtrap.models.train --model pretrained
#
# Output: models/<model>.keras             the trained model
#         models/<model>.json              best settings, validation score, class names
#         reports/results/optuna_<model>.csv  every Optuna trial
#         reports/results/history_<model>.csv training curves of the final model
#         reports/figures/optuna_<model>.png  optimisation history and parameter importance
#         reports/figures/training_<model>.png
# =====================================================================

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from camtrap.data.download import load_config


# Step 1: Load the photos of one split. Photos are read once and kept in memory as whole
#         numbers (uint8, a quarter of the memory of decimals). Training photos are
#         reshuffled every epoch; validation photos keep a fixed order.
def load_split(folder, size, batch_size, training, seed):
    ds = keras.utils.image_dataset_from_directory(folder, image_size=(size, size),
                                                  batch_size=None, shuffle=False)
    class_names = ds.class_names
    ds = ds.map(lambda x, y: (tf.cast(x, tf.uint8), y)).cache()
    if training:
        ds = ds.shuffle(10_000, seed=seed)
    ds = ds.batch(batch_size).map(lambda x, y: (tf.cast(x, tf.float32), y))
    return ds.prefetch(tf.data.AUTOTUNE), class_names


# Step 2: The two models
def augment(x):
    # Random changes applied only while training, so the model sees more variety
    x = layers.RandomFlip("horizontal")(x)
    x = layers.RandomBrightness(0.2)(x)      # darker/lighter photos, e.g. dusk
    x = layers.RandomContrast(0.2)(x)
    return x


def build_scratch(params, n_classes, size):
    inputs = keras.Input(shape=(size, size, 3))
    x = augment(inputs) if params["augment"] else inputs
    x = layers.Rescaling(1.0 / 255)(x)                   # pixels from [0, 255] to [0, 1]
    for filters in [16, 32, 64, 128]:                    # each block finds larger patterns
        x = layers.Conv2D(filters, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(params["dense_units"], activation="relu")(x)
    x = layers.Dropout(params["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="scratch_cnn")


def build_pretrained(params, n_classes, size):
    base = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                             input_shape=(size, size, 3), pooling="avg")
    # Freeze the network, then let the last `unfreeze_layers` layers learn (0 = fully frozen)
    base.trainable = params["unfreeze_layers"] > 0
    for layer in base.layers[:len(base.layers) - params["unfreeze_layers"]]:
        layer.trainable = False

    inputs = keras.Input(shape=(size, size, 3))         # EfficientNet scales the pixels itself
    x = augment(inputs) if params["augment"] else inputs
    x = base(x, training=False)                          # keeps the batch-normalisation statistics fixed
    x = layers.Dropout(params["dropout"])(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="pretrained_efficientnetb0")


BUILDERS = {"scratch": build_scratch, "pretrained": build_pretrained}


# Step 3: The settings Optuna is allowed to try
def suggest_params(trial, model_name):
    if model_name == "scratch":
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "dense_units": trial.suggest_categorical("dense_units", [64, 128, 256]),
        }
    else:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "unfreeze_layers": trial.suggest_categorical("unfreeze_layers", [0, 20, 50]),
        }
    params["dropout"] = trial.suggest_float("dropout", 0.2, 0.5)
    params["augment"] = trial.suggest_categorical("augment", [True, False])
    return params


def compile_model(model_name, params, n_classes, size):
    model = BUILDERS[model_name](params, n_classes, size)
    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def macro_f1(model, ds, y_true):
    y_pred = model.predict(ds, verbose=0).argmax(axis=1)
    return f1_score(y_true, y_pred, average="macro")


# Lets Optuna stop a trial early if it is doing worse than the median trial (MedianPruner).
# Validation accuracy is used as the epoch-by-epoch signal because Keras computes it for free;
# the final score of each trial is macro-F1.
class ReportToOptuna(keras.callbacks.Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(logs["val_accuracy"], step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# Step 4: Plots for the report (same two plots as Lab 4)
def plot_study(study, model_name, path):
    trials = study.trials_dataframe()
    trials = trials[trials["state"] == "COMPLETE"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(trials["number"], trials["value"], color="steelblue", label="Trial")
    axes[0].plot(trials["number"], trials["value"].cummax(), color="red", lw=2, label="Best so far")
    axes[0].set_xlabel("Trial number")
    axes[0].set_ylabel("Validation macro-F1")
    axes[0].set_title(f"Optimisation history ({model_name})")
    axes[0].legend()

    try:
        importances = optuna.importance.get_param_importances(study)
        names, values = zip(*sorted(importances.items(), key=lambda kv: kv[1]))
        axes[1].barh(names, values, color="steelblue")
        axes[1].set_xlabel("Importance (fANOVA)")
    except (RuntimeError, ValueError):
        axes[1].text(0.5, 0.5, "Not enough completed trials", ha="center")
    axes[1].set_title("Hyperparameter importance")

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_history(history, model_name, path):
    h = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.plot(h[metric], label="Train")
        ax.plot(h[f"val_{metric}"], label="Validation")
        ax.set_title(f"{metric.capitalize()} ({model_name})")
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(BUILDERS), required=True)
    parser.add_argument("--quick", action="store_true", help="2 one-epoch trials, to check that everything runs")
    args = parser.parse_args()
    model_name = args.model

    config = load_config()
    cfg = config["train"]
    if args.quick:
        cfg = {**cfg, "n_trials": {model_name: 2}, "trial_epochs": 1, "max_epochs": 1}
    seed, size = config["seed"], config["data"]["image_size"]
    images = Path(config["paths"]["images"])
    models_dir = Path(config["paths"]["models"])
    reports = Path(config["paths"]["reports"])
    for folder in [models_dir, reports / "results", reports / "figures"]:
        folder.mkdir(parents=True, exist_ok=True)

    keras.utils.set_random_seed(seed)
    train_ds, class_names = load_split(images / "train", size, cfg["batch_size"], training=True, seed=seed)
    val_ds, _ = load_split(images / "val", size, cfg["batch_size"], training=False, seed=seed)
    n_classes = len(class_names)
    y_train = np.concatenate([y.numpy() for _, y in train_ds])
    y_val = np.concatenate([y.numpy() for _, y in val_ds])

    # Rare classes count more in the loss, so every class matters equally (as in Lab 3)
    weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y_train)
    class_weight = dict(enumerate(weights))

    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["mlflow_experiment"])

    # Step 5: One Optuna trial = train with one set of settings, return validation macro-F1
    def objective(trial):
        params = suggest_params(trial, model_name)
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        model = compile_model(model_name, params, n_classes, size)
        model.fit(train_ds, validation_data=val_ds, epochs=cfg["trial_epochs"],
                  class_weight=class_weight, callbacks=[ReportToOptuna(trial)], verbose=0)
        score = macro_f1(model, val_ds, y_val)

        with mlflow.start_run(run_name=f"{model_name}-trial-{trial.number}"):
            mlflow.log_params({"model": model_name, **params})
            mlflow.log_metric("val_macro_f1", score)
        print(f"Trial {trial.number}: macro-F1 {score:.3f}  {params}")
        return score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=cfg["n_trials"][model_name])

    study.trials_dataframe().to_csv(reports / "results" / f"optuna_{model_name}.csv", index=False)
    plot_study(study, model_name, reports / "figures" / f"optuna_{model_name}.png")
    print(f"\nBest trial #{study.best_trial.number}: macro-F1 {study.best_value:.3f}")

    # Step 6: Retrain the best settings for longer, with early stopping on the validation loss
    best = dict(study.best_params)
    keras.backend.clear_session()
    keras.utils.set_random_seed(seed)
    model = compile_model(model_name, best, n_classes, size)
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"],
                                               restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=cfg["max_epochs"],
                        class_weight=class_weight, callbacks=[early_stop], verbose=2)
    final_f1 = macro_f1(model, val_ds, y_val)

    # Step 7: Save the model and a summary; the quality gate decides if the API may use it
    passed = final_f1 >= cfg["quality_gate_macro_f1"]
    model.save(models_dir / f"{model_name}.keras")
    summary = {"model": model_name, "params": best, "val_macro_f1": round(float(final_f1), 4),
               "passed_quality_gate": bool(passed), "epochs_trained": len(history.history["loss"]),
               "class_names": class_names, "image_size": size}
    (models_dir / f"{model_name}.json").write_text(json.dumps(summary, indent=2))

    pd.DataFrame(history.history).to_csv(reports / "results" / f"history_{model_name}.csv", index_label="epoch")
    plot_history(history, model_name, reports / "figures" / f"training_{model_name}.png")

    with mlflow.start_run(run_name=f"{model_name}-final"):
        mlflow.log_params({"model": model_name, **best})
        mlflow.log_metric("val_macro_f1", final_f1)

    print(f"\nFinal {model_name} model: validation macro-F1 {final_f1:.3f} "
          f"({'passed' if passed else 'FAILED'} the quality gate of {cfg['quality_gate_macro_f1']})")


if __name__ == "__main__":
    main()
