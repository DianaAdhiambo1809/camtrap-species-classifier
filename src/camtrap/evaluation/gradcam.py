# =====================================================================
# Grad-CAM: where in the photo does the final model look?
# =====================================================================
# Grad-CAM colours the parts of a photo that pushed the model towards its answer.
# It is computed from the last convolution layer of EfficientNetB0 (a 7 x 7 grid)
# and stretched over the 224 x 224 photo.
#
# Every Snapshot Serengeti photo has a white strip at the bottom with the camera
# brand and the date and time. It could be a shortcut: the time hints at night
# animals, and the brand text differs slightly between cameras. Two checks:
#   A. the share of the Grad-CAM heat that falls on the strip
#      (6.7% would be expected if the heat were spread evenly)
#   B. paint the strip grey on every test photo and compare the predictions
#
# Needs TensorFlow, the photos and models/pretrained.keras, so run it in Colab:
#   python -m camtrap.evaluation.gradcam
#
# Output: reports/figures/gradcam_correct.png, gradcam_wrong.png
#         reports/results/gradcam_strip_check.json
# =====================================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras

from camtrap.data.download import load_config


# Step 1: Split the model into (photo -> last convolution layer) and the final layer
def gradcam_parts(model):
    base = next(layer for layer in model.layers if isinstance(layer, keras.Model))   # EfficientNetB0
    features = keras.Model(base.inputs, [base.get_layer("top_activation").output, base.output])
    head = model.layers[-1]                                                           # final Dense layer
    return features, head


# Step 2: Grad-CAM for a batch of photos, for the class the model predicted
def gradcam(features, head, images):
    with tf.GradientTape() as tape:
        conv, pooled = features(images, training=False)
        logits = tf.matmul(pooled, head.kernel) + head.bias      # scores before softmax
        predicted = tf.argmax(logits, axis=1)
        score = tf.gather(logits, predicted, batch_dims=1)
    grads = tape.gradient(score, conv)

    weights = tf.reduce_mean(grads, axis=(1, 2))                  # importance of each feature map
    cam = tf.nn.relu(tf.reduce_sum(conv * weights[:, None, None, :], axis=-1))
    cam = cam / (tf.reduce_max(cam, axis=(1, 2), keepdims=True) + 1e-8)
    return cam.numpy(), tf.nn.softmax(logits).numpy()


def stretch(cam, size):
    return tf.image.resize(cam[..., None], (size, size)).numpy()[..., 0]


# Step 3: The same photos with the bottom strip painted grey
def blank_strip(images, rows, size):
    keep = tf.concat([tf.ones(size - rows), tf.zeros(rows)], axis=0)[None, :, None, None]
    return images * keep + 128.0 * (1 - keep)


def plot_examples(rows, cams, size, title, path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    for ax, (_, row) in zip(axes.ravel(), rows.iterrows()):
        photo = np.asarray(Image.open(row["path"]).convert("RGB").resize((size, size)))
        ax.imshow(photo)
        ax.imshow(stretch(cams[row.name], size), cmap="jet", alpha=0.4)
        ax.set_title(f"true: {row['true']}\npredicted: {row['pred']} ({row['confidence']:.0%})", fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    config = load_config()
    size = config["data"]["image_size"]
    strip_rows = config["gradcam"]["strip_rows"]
    models_dir = Path(config["paths"]["models"])
    results = Path(config["paths"]["reports"]) / "results"
    figures = Path(config["paths"]["reports"]) / "figures"

    info = json.loads((models_dir / "pretrained.json").read_text())
    class_names = info["class_names"]
    model = keras.models.load_model(models_dir / "pretrained.keras")
    features, head = gradcam_parts(model)

    ds = keras.utils.image_dataset_from_directory(Path(config["paths"]["images"]) / "test",
                                                  image_size=(size, size), batch_size=32, shuffle=False)
    assert ds.class_names == class_names, "class order differs from training"

    # Step 4: Grad-CAM and the blanked-strip predictions for every test photo
    cams, probs, probs_blank, labels = [], [], [], []
    for images, y in ds:
        cam, p = gradcam(features, head, images)
        cams.append(cam)
        probs.append(p)
        probs_blank.append(model.predict(blank_strip(images, strip_rows, size), verbose=0))
        labels.append(y.numpy())
    cams, probs, probs_blank = np.concatenate(cams), np.concatenate(probs), np.concatenate(probs_blank)

    # Check that the split model gives the same answers as the whole model
    first = next(iter(ds))[0]
    assert np.allclose(probs[:len(first)], model.predict(first, verbose=0), atol=1e-3)

    test = pd.DataFrame({"path": ds.file_paths,
                         "true": [class_names[i] for i in np.concatenate(labels)],
                         "pred": [class_names[i] for i in probs.argmax(axis=1)],
                         "pred_blank": [class_names[i] for i in probs_blank.argmax(axis=1)],
                         "confidence": probs.max(axis=1)})
    test["correct"] = test["true"] == test["pred"]

    # Check A: share of the heat on the strip (stretched to the full photo size)
    heat = np.stack([stretch(c, size) for c in cams])
    test["strip_share"] = heat[:, -strip_rows:, :].sum(axis=(1, 2)) / (heat.sum(axis=(1, 2)) + 1e-8)

    check = {
        "strip_rows": strip_rows,
        "A_gradcam_share_on_strip": {
            "expected_if_spread_evenly": round(strip_rows / size, 4),
            "all_photos": round(float(test["strip_share"].mean()), 4),
            "correct_photos": round(float(test.loc[test["correct"], "strip_share"].mean()), 4),
            "wrong_photos": round(float(test.loc[~test["correct"], "strip_share"].mean()), 4),
        },
        "B_strip_painted_grey": {
            "macro_f1_original": round(f1_score(test["true"], test["pred"], average="macro"), 4),
            "macro_f1_blanked": round(f1_score(test["true"], test["pred_blank"], average="macro"), 4),
            "accuracy_original": round(accuracy_score(test["true"], test["pred"]), 4),
            "accuracy_blanked": round(accuracy_score(test["true"], test["pred_blank"]), 4),
            "share_of_answers_changed": round(float((test["pred"] != test["pred_blank"]).mean()), 4),
        },
    }
    (results / "gradcam_strip_check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps(check, indent=2))

    # Step 5: Example pictures: the most confident correct photo of 8 classes, and the 8 most confident mistakes
    correct = (test[test["correct"]].sort_values("confidence", ascending=False)
               .groupby("true").head(1).head(8))
    wrong = test[~test["correct"]].sort_values("confidence", ascending=False).head(8)
    plot_examples(correct, cams, size, "Grad-CAM, correct answers (test cameras)", figures / "gradcam_correct.png")
    plot_examples(wrong, cams, size, "Grad-CAM, most confident mistakes (test cameras)", figures / "gradcam_wrong.png")
    print("Saved gradcam_correct.png and gradcam_wrong.png")


if __name__ == "__main__":
    main()
