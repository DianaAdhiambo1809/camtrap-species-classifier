# =====================================================================
# Predict the validation and test photos with every trained CNN
# =====================================================================
# Saves the class probabilities of each photo, so the evaluation
# (camtrap.evaluation.evaluate) can run on any computer without TensorFlow.
#
# Needs TensorFlow, the photos and the trained models, so run it in Colab:
#   python -m camtrap.models.predict
#
# Output: reports/results/predictions_<model>.csv
#         one row per photo: file, split, true class, predicted class, prob_<class> ...
# =====================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from camtrap.data.download import load_config


def main():
    config = load_config()
    size = config["data"]["image_size"]
    images = Path(config["paths"]["images"])
    models_dir = Path(config["paths"]["models"])
    results = Path(config["paths"]["reports"]) / "results"
    results.mkdir(parents=True, exist_ok=True)

    for model_path in sorted(models_dir.glob("*.keras")):
        name = model_path.stem
        class_names = json.loads((models_dir / f"{name}.json").read_text())["class_names"]
        model = keras.models.load_model(model_path)

        parts = []
        for split in ["val", "test"]:
            ds = keras.utils.image_dataset_from_directory(images / split, image_size=(size, size),
                                                          batch_size=64, shuffle=False)
            assert ds.class_names == class_names, "class order differs from training"

            probs = model.predict(ds, verbose=0)
            y = np.concatenate([labels.numpy() for _, labels in ds])

            part = pd.DataFrame(probs, columns=[f"prob_{c}" for c in class_names])
            part.insert(0, "file", [Path(p).name for p in ds.file_paths])
            part.insert(1, "split", split)
            part.insert(2, "true", [class_names[i] for i in y])
            part.insert(3, "pred", [class_names[i] for i in probs.argmax(axis=1)])
            parts.append(part)

        pd.concat(parts).to_csv(results / f"predictions_{name}.csv", index=False)
        print(f"{name}: saved predictions for {sum(len(p) for p in parts):,} photos")


if __name__ == "__main__":
    main()
