# =====================================================================
# Split the cameras into train / validation / test and pick the photo sample
# =====================================================================
# Whole cameras go to one split only, so the test cameras are never seen
# in training (audit Steps 5 and 8). Then a balanced sample of bursts is
# picked from each split, with a limit per camera.
#
# How to run it (from the project folder):
#   python -m camtrap.data.split
#
# Needs:  data/processed/bursts.csv (made by clean.py)
# Output: data/processed/bursts_split.csv  every clean burst with its split
#         data/processed/sample.csv        the photos to download
# =====================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from camtrap.data.download import load_config


# Step 1: Go through the cameras in a random order. Fill the test split first, then
#         validation, until each holds about its share of the animal bursts; the rest is train.
def assign_cameras(bursts, val_share, test_share, seed):
    animal_bursts = bursts[bursts["class"] != "empty"].groupby("location").size()
    total = animal_bursts.sum()

    cameras = np.sort(bursts["location"].unique())    # sort first, so the shuffle is repeatable
    order = np.random.default_rng(seed).permutation(cameras)

    split_of = {}
    in_test = in_val = 0
    for camera in order:
        n = animal_bursts.get(camera, 0)
        if in_test < test_share * total:
            split_of[camera] = "test"
            in_test += n
        elif in_val < val_share * total:
            split_of[camera] = "val"
            in_val += n
        else:
            split_of[camera] = "train"
    return split_of


# Step 2: A split is only usable if every class has enough bursts in validation and test
def every_class_covered(bursts, min_bursts):
    counts = (bursts[bursts["split"] != "train"]
              .groupby(["split", "class"]).size()
              .unstack(fill_value=0)
              .reindex(index=["val", "test"], columns=bursts["class"].unique(), fill_value=0))
    return (counts >= min_bursts).all().all()


# Step 3: Pick the sample: at most `per_camera` bursts of a class from any one camera,
#         then at most `per_class[split]` bursts of each class in each split
def pick_sample(bursts, per_class, per_camera, seed):
    picked = []
    for (split, cls), group in bursts.groupby(["split", "class"]):
        group = group.sample(frac=1, random_state=seed)           # shuffle
        group = group.groupby("location").head(per_camera)        # limit per camera
        n = min(len(group), per_class[split])
        picked.append(group.head(n))
    return pd.concat(picked, ignore_index=True)


def main():
    config = load_config()
    processed = Path(config["paths"]["processed"])
    split_cfg, sample_cfg = config["split"], config["sample"]

    bursts = pd.read_csv(processed / "bursts.csv")

    # Try camera orders until every class appears in validation and test
    for attempt in range(split_cfg["max_tries"]):
        seed = config["seed"] + attempt
        split_of = assign_cameras(bursts, split_cfg["val_share"], split_cfg["test_share"], seed)
        bursts["split"] = bursts["location"].map(split_of)
        if every_class_covered(bursts, split_cfg["min_class_bursts"]):
            break
    else:
        raise RuntimeError("No camera split gave every class enough bursts; lower min_class_bursts")
    print(f"Camera split found with seed {seed} (attempt {attempt + 1})")

    # Safety check: no camera may appear in two splits
    assert bursts.groupby("location")["split"].nunique().max() == 1

    sample = pick_sample(bursts, sample_cfg["per_class"], sample_cfg["per_camera_per_class"], config["seed"])
    sample["url"] = sample_cfg["image_base_url"] + "/" + sample["photo"]

    bursts.to_csv(processed / "bursts_split.csv", index=False)
    sample.to_csv(processed / "sample.csv", index=False)

    # Summary
    print("\nCameras per split:")
    print(bursts.groupby("split")["location"].nunique().to_string())
    print("\nShare of animal bursts per split:")
    print(bursts[bursts["class"] != "empty"]["split"].value_counts(normalize=True).round(3).to_string())
    print("\nSample (photos to download), class x split:")
    table = pd.crosstab(sample["class"], sample["split"], margins=True, margins_name="total")
    print(table[["train", "val", "test", "total"]].to_string())
    print(f"\nSaved {len(sample):,} photos to {processed / 'sample.csv'}")


if __name__ == "__main__":
    main()
