# =====================================================================
# Baseline models: the scores any real model has to beat
# =====================================================================
# 1. Dummy models, which ignore the photo completely. Scored on all validation
#    bursts (the real mix, mostly empty) and on the balanced validation sample.
#    They show why accuracy is misleading for this data.
# 2. Logistic regression on pretrained image features (from features.py).
#    The regularisation strength C is chosen by cross-validation that keeps
#    whole cameras together (GroupKFold).
# 3. The same model scored with a random split and with a camera split,
#    to measure how much a random split would overstate performance.
#
# How to run it (from the project folder):
#   python -m camtrap.models.baselines
#
# Needs:  data/processed/bursts_split.csv and sample.csv;
#         data/processed/features.npz for parts 2 and 3
# Output: reports/results/baselines.csv
#         reports/figures/baseline_confusion_matrix.png
#         reports/results/predictions_logreg.csv (val and test, for evaluate.py)
# =====================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from camtrap.data.download import load_config


def score(model, data, y_true, y_pred):
    return {"model": model, "data": data,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def logistic_model(C):
    # Features are scaled first; class_weight="balanced" so rare classes count as much as common ones
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


# Part 1: Models that ignore the photo
def dummy_baselines(train, val, data_name, seed):
    rows = []
    for strategy in ["most_frequent", "stratified"]:
        dummy = DummyClassifier(strategy=strategy, random_state=seed)
        dummy.fit(np.zeros((len(train), 1)), train["class"])      # the input is ignored
        pred = dummy.predict(np.zeros((len(val), 1)))
        rows.append(score(f"dummy ({strategy})", data_name, val["class"], pred))
    return rows


# Load the features and look up the camera of each photo in sample.csv
def load_features(processed):
    f = np.load(processed / "features.npz")
    sample = pd.read_csv(processed / "sample.csv")
    camera_of = dict(zip(sample["photo"].apply(lambda p: Path(p).stem + ".jpg"), sample["location"]))

    data, files = {}, {}
    for split in ["train", "val", "test"]:
        files[split] = f[f"files_{split}"]
        cameras = np.array([camera_of[name] for name in files[split]])
        data[split] = (f[f"X_{split}"], f[f"y_{split}"], cameras)
    return data, files, list(f["class_names"])


# Part 2: Logistic regression, with C chosen by camera-grouped cross-validation
def logistic_baseline(data, class_names, C_values):
    X_train, y_train, cameras = data["train"]
    X_val, y_val, _ = data["val"]

    cv_scores = {}
    for C in C_values:
        cv = cross_validate(logistic_model(C), X_train, y_train, groups=cameras,
                            cv=GroupKFold(n_splits=5), scoring="f1_macro")
        cv_scores[C] = cv["test_score"].mean()
        print(f"  C = {C:<6}  camera-grouped CV macro-F1: {cv_scores[C]:.3f}")
    best_C = max(cv_scores, key=cv_scores.get)
    print(f"Best C: {best_C}")

    model = logistic_model(best_C).fit(X_train, y_train)
    pred = model.predict(X_val)
    print("\nValidation sample:")
    print(classification_report(y_val, pred, target_names=class_names, digits=3))
    return model, pred, best_C


# Save validation and test predictions in the same format as predict.py, for evaluate.py
def save_predictions(model, data, files, class_names, path):
    parts = []
    for split in ["val", "test"]:
        X, y, _ = data[split]
        probs = model.predict_proba(X)
        part = pd.DataFrame(probs, columns=[f"prob_{c}" for c in class_names])
        part.insert(0, "file", files[split])
        part.insert(1, "split", split)
        part.insert(2, "true", [class_names[i] for i in y])
        part.insert(3, "pred", [class_names[i] for i in probs.argmax(axis=1)])
        parts.append(part)
    pd.concat(parts).to_csv(path, index=False)


# Part 3: The same model, scored with a random split and with a camera split
def random_vs_camera_split(data, C, seed):
    X, y, cameras = data["train"]
    splits = {
        "train CV, random split": (StratifiedKFold(n_splits=5, shuffle=True, random_state=seed), None),
        "train CV, camera split": (GroupKFold(n_splits=5), cameras),
    }
    rows = []
    for name, (cv, groups) in splits.items():
        result = cross_validate(logistic_model(C), X, y, groups=groups, cv=cv,
                                scoring=["accuracy", "f1_macro"])
        rows.append({"model": "logistic regression", "data": name,
                     "accuracy": result["test_accuracy"].mean(),
                     "macro_f1": result["test_f1_macro"].mean()})
    return rows


def main():
    config = load_config()
    seed = config["seed"]
    processed = Path(config["paths"]["processed"])
    reports = Path(config["paths"]["reports"])
    (reports / "results").mkdir(parents=True, exist_ok=True)
    (reports / "figures").mkdir(parents=True, exist_ok=True)

    bursts = pd.read_csv(processed / "bursts_split.csv")
    sample = pd.read_csv(processed / "sample.csv")

    rows = dummy_baselines(bursts[bursts["split"] == "train"], bursts[bursts["split"] == "val"],
                           "all val bursts", seed)
    rows += dummy_baselines(sample[sample["split"] == "train"], sample[sample["split"] == "val"],
                            "val sample", seed)

    if (processed / "features.npz").exists():
        data, files, class_names = load_features(processed)
        model, pred, best_C = logistic_baseline(data, class_names, config["baseline"]["C_values"])
        rows.append(score("logistic regression", "val sample", data["val"][1], pred))
        save_predictions(model, data, files, class_names, reports / "results" / "predictions_logreg.csv")
        rows += random_vs_camera_split(data, best_C, seed)

        ConfusionMatrixDisplay.from_predictions(data["val"][1], pred, display_labels=class_names,
                                                normalize="true", values_format=".2f",
                                                xticks_rotation=90, cmap="Blues")
        plt.title("Logistic regression baseline, validation cameras\n(rows = true class, share of each row)")
        plt.gcf().set_size_inches(10, 9)
        plt.tight_layout()
        plt.savefig(reports / "figures" / "baseline_confusion_matrix.png", dpi=120)
    else:
        print("features.npz not found: run `python -m camtrap.models.features` in Colab first.")
        print("Only the dummy baselines were run.\n")

    results = pd.DataFrame(rows)
    results.to_csv(reports / "results" / "baselines.csv", index=False)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
