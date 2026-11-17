# =====================================================================
# Turn every photo into a feature vector with a pretrained network
# =====================================================================
# EfficientNetB0, trained on ImageNet, is used without its last layer:
# each 224x224 photo becomes 1,280 numbers describing what it shows.
# The features feed the logistic regression baseline (baselines.py).
#
# Needs TensorFlow and the photos in data/images/, so run it in Colab:
#   python -m camtrap.models.features
#
# Output: data/processed/features.npz
# =====================================================================

from pathlib import Path

import numpy as np
from tensorflow import keras

from camtrap.data.download import load_config

SPLITS = ["train", "val", "test"]


def main():
    config = load_config()
    size = config["data"]["image_size"]
    images = Path(config["paths"]["images"])
    processed = Path(config["paths"]["processed"])

    # Step 1: The pretrained network without its classification layer.
    # pooling="avg" averages the last feature maps into one vector per photo.
    model = keras.applications.EfficientNetB0(include_top=False, weights="imagenet",
                                              pooling="avg", input_shape=(size, size, 3))

    arrays = {}
    for split in SPLITS:
        # Step 2: Load one split; the class is the sub-folder name. shuffle=False keeps the order fixed.
        ds = keras.utils.image_dataset_from_directory(images / split, image_size=(size, size),
                                                      batch_size=64, shuffle=False)

        # Step 3: Run every photo through the network
        X = model.predict(ds, verbose=1)
        y = np.concatenate([labels.numpy() for _, labels in ds])

        arrays[f"X_{split}"] = X.astype("float32")
        arrays[f"y_{split}"] = y
        arrays[f"files_{split}"] = np.array([Path(p).name for p in ds.file_paths])
        print(f"{split}: {X.shape[0]:,} photos -> {X.shape[1]} features each")

    arrays["class_names"] = np.array(ds.class_names)
    np.savez_compressed(processed / "features.npz", **arrays)
    print(f"Saved {processed / 'features.npz'}")


if __name__ == "__main__":
    main()
