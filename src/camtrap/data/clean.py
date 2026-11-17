# =====================================================================
# Clean the Snapshot Serengeti labels
# =====================================================================
# Turns the raw labels into one tidy table with one row per burst.
# The rules come from the data audit (notebooks/01_data_audit.ipynb, Step 12).
#
# How to run it (from the project folder):
#   python -m camtrap.data.clean
#
# Output: data/processed/bursts.csv
# =====================================================================

import json
from pathlib import Path

import pandas as pd

from camtrap.data.download import load_config

# Columns that are only filled in when there is an animal (blank for empty bursts),
# so a model could use them to "cheat". Audit Step 10.
BEHAVIOUR_COLUMNS = ["count", "standing", "resting", "moving", "interacting", "young_present"]

# Lions are labelled by sex and age; for species classification they are one class. Audit Step 6.
LION_LABELS = {"lionfemale": "lion", "lionmale": "lion", "lioncub": "lion"}


# Step 1: Read the labels file of every season in the config
def load_labels(config):
    raw_folder = Path(config["paths"]["raw"])
    all_photos, all_labels = [], []

    for season in config["data"]["seasons"]:
        with open(raw_folder / f"SnapshotSerengeti{season}.json") as f:
            data = json.load(f)

        names = {c["id"]: c["name"] for c in data["categories"]}    # 0 -> "empty", 5 -> "zebra", ...
        labels = pd.DataFrame(data["annotations"])
        labels["label"] = labels["category_id"].map(names)

        all_photos.append(pd.DataFrame(data["images"]))
        all_labels.append(labels)

    photos = pd.concat(all_photos, ignore_index=True)
    labels = pd.concat(all_labels, ignore_index=True)
    print(f"Loaded {len(photos):,} photos and {len(labels):,} labels")
    return photos, labels


# Step 2: Build one row per burst (labels describe the whole burst, not each photo)
def build_bursts(photos, labels):
    labels = labels.drop(columns=BEHAVIOUR_COLUMNS)
    labels["label"] = labels["label"].replace(LION_LABELS)

    photos["datetime"] = pd.to_datetime(photos["datetime"], errors="coerce")
    photos = photos.sort_values(["seq_id", "frame_num"])

    bursts = photos.groupby("seq_id").agg(
        location=("location", "first"),
        start=("datetime", "min"),
        n_photos=("id", "size"),
        # the middle photo of the burst is the one most likely to show the animal
        photo=("file_name", lambda x: x.iloc[len(x) // 2]),
    )
    bursts["season"] = labels.groupby("seq_id")["season"].first()
    bursts["labels"] = labels.groupby("seq_id")["label"].apply(lambda x: sorted(set(x)))
    return bursts.reset_index()


# Step 3: Remove bursts with people (privacy) and bursts with more than one species
def remove_bursts(bursts):
    has_people = bursts["labels"].apply(lambda x: "human" in x)
    print(f"Removed {has_people.sum():,} bursts with people")
    bursts = bursts[~has_people]

    mixed = bursts["labels"].apply(len) > 1
    print(f"Removed {mixed.sum():,} bursts with more than one species")
    bursts = bursts[~mixed].copy()

    bursts["label"] = bursts["labels"].str[0]     # exactly one label is left in each burst
    return bursts.drop(columns="labels")


# Step 4: Choose the species that become classes; every other animal becomes "other"
def assign_classes(bursts, config):
    rules = config["classes"]

    animals = bursts[(bursts["label"] != "empty") & (~bursts["label"].isin(rules["group_labels"]))]
    n_bursts = animals["label"].value_counts()

    # number of cameras where the species was seen regularly
    per_camera = animals.groupby(["label", "location"]).size()
    regular_cameras = (per_camera >= rules["regular_camera_bursts"]).groupby("label").sum()
    regular_cameras = regular_cameras.reindex(n_bursts.index)

    keep = n_bursts[(n_bursts >= rules["min_bursts"]) &
                    (regular_cameras >= rules["min_regular_cameras"])].index

    bursts["class"] = "other"
    bursts.loc[bursts["label"].isin(keep), "class"] = bursts["label"]
    bursts.loc[bursts["label"] == "empty", "class"] = "empty"

    print(f"\n{len(keep)} species kept as classes: {', '.join(sorted(keep))}")
    return bursts


# Step 5: Tag each burst as day or night; bursts with wrong dates are "unknown"
def add_time_of_day(bursts, config):
    night_start, night_end = config["data"]["night_hours"]

    valid = pd.Series(False, index=bursts.index)
    for season, (first_day, last_day) in config["data"]["valid_dates"].items():
        in_range = bursts["start"].between(first_day, last_day)
        valid = valid | ((bursts["season"] == season) & in_range)

    hour = bursts["start"].dt.hour
    bursts["time_of_day"] = "day"
    bursts.loc[(hour >= night_start) | (hour < night_end), "time_of_day"] = "night"
    bursts.loc[~valid, "time_of_day"] = "unknown"

    print(f"Bursts with an invalid date (time of day unknown): {(~valid).sum()}")
    return bursts


def main():
    config = load_config()

    photos, labels = load_labels(config)
    bursts = build_bursts(photos, labels)
    bursts = remove_bursts(bursts)
    bursts = assign_classes(bursts, config)
    bursts = add_time_of_day(bursts, config)

    # Step 6: Save the clean table
    out_folder = Path(config["paths"]["processed"])
    out_folder.mkdir(parents=True, exist_ok=True)
    columns = ["seq_id", "season", "location", "start", "time_of_day", "n_photos", "photo", "label", "class"]
    bursts[columns].to_csv(out_folder / "bursts.csv", index=False)

    print(f"\nSaved {len(bursts):,} bursts to {out_folder / 'bursts.csv'}")
    print("\nBursts per class:")
    print(bursts["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
