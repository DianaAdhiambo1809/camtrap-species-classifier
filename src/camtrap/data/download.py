# =====================================================================
# Download the Snapshot Serengeti labels (metadata)
# =====================================================================
# What this file does:
#   The dataset has two parts: the photos, and a labels file that says,
#   for every photo, which camera took it, when, and which animal is in it.
#   We download the labels first, because they tell us which photos we need.
#
# How to run it (from the project folder):
#   python src/camtrap/data/download.py
# =====================================================================

# Step 1: Import the libraries we need
import zipfile               # to open .zip files
from pathlib import Path     # to build folder paths that work on Windows and Linux

import requests              # to download files from the internet
import yaml                  # to read our settings file (config.yaml)


# Step 2: Where the labels files are stored online (LILA BC's Google Cloud copy)
BASE_URL = "https://storage.googleapis.com/public-datasets-lila/snapshotserengeti-v-2-0"


# Step 3: Read the settings from config.yaml
def load_config(path="configs/config.yaml"):
    with open(path) as f:
        config = yaml.safe_load(f)    # turn the YAML text into a Python dictionary
    return config


# Step 4: Download one file, trying again if the internet connection drops
def download_file(url, save_to, max_tries=5):
    for attempt in range(1, max_tries + 1):
        try:
            response = requests.get(url, stream=True, timeout=60)   # stream = download in pieces
            response.raise_for_status()                             # stop if the server sends an error (e.g. 404)

            with open(save_to, "wb") as f:                          # "wb" = write bytes (a zip is not text)
                for piece in response.iter_content(chunk_size=1024 * 1024):   # 1 MB at a time
                    f.write(piece)
            return                                                  # success, so leave the function

        except requests.RequestException as error:
            print(f"  Attempt {attempt} failed: {error}")

    # If we reach this line, every attempt failed
    raise RuntimeError(f"Could not download {url} after {max_tries} attempts")


# Step 5: Download and unzip the labels for each season listed in config.yaml
def download_labels(config):
    raw_folder = Path(config["paths"]["raw"])        # data/raw
    raw_folder.mkdir(parents=True, exist_ok=True)    # create the folder if it does not exist yet

    for season in config["data"]["seasons"]:         # e.g. ["S01"]
        json_file = raw_folder / f"SnapshotSerengeti{season}.json"
        zip_file = raw_folder / f"SnapshotSerengeti{season}.json.zip"

        # Skip seasons we already have, so running the script twice does not download twice
        if json_file.exists():
            print(f"{json_file} already exists, skipping")
            continue

        print(f"Downloading labels for season {season} (about 20 MB) ...")
        download_file(f"{BASE_URL}/SnapshotSerengeti{season}.json.zip", zip_file)

        # Unzip, then delete the zip file to save disk space
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(raw_folder)
        zip_file.unlink()

        print(f"Saved {json_file}")


# Step 6: Run everything when this file is executed
if __name__ == "__main__":
    config = load_config()
    download_labels(config)
