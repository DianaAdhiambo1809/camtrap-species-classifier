# =====================================================================
# Download the photo sample and shrink it
# =====================================================================
# Reads data/processed/sample.csv (made by split.py), downloads each photo,
# resizes it and saves it as
#     data/images/<split>/<class>/<photo name>.jpg
# This is the folder layout keras.utils.image_dataset_from_directory expects.
#
# Photos already on disk are skipped, so the script can be stopped and rerun.
# The sample is about 8,000 photos (~5 GB before resizing), so run it in
# Colab, where the download is fast.
#
# How to run it (from the project folder):
#   python -m camtrap.data.download_images
# =====================================================================

import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

from camtrap.data.download import load_config


# Step 1: Where each photo is saved
def image_path(row, images_folder):
    return Path(images_folder) / row["split"] / row["class"] / (Path(row["photo"]).stem + ".jpg")


# Step 2: Download one photo, resize it and save it.
#         Returns None if it worked, otherwise the reason it failed.
def fetch_and_resize(url, save_to, size, max_tries=3):
    for attempt in range(1, max_tries + 1):
        try:
            response = requests.get(url, timeout=60)
            if response.status_code == 404:
                return "not found"                  # trying again will not help
            response.raise_for_status()
            break
        except requests.RequestException as error:
            if attempt == max_tries:
                return f"download failed: {error}"

    try:
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
    except OSError:
        return "not a readable image"

    image = image.resize((size, size))              # 2048x1536 -> size x size
    save_to.parent.mkdir(parents=True, exist_ok=True)
    image.save(save_to, quality=90)
    return None


def main():
    config = load_config()
    size = config["data"]["image_size"]
    processed = Path(config["paths"]["processed"])

    sample = pd.read_csv(processed / "sample.csv")
    sample["image_path"] = [image_path(row, config["paths"]["images"]) for _, row in sample.iterrows()]

    # Step 3: Skip the photos we already have
    todo = sample[~sample["image_path"].apply(lambda p: p.exists())]
    print(f"{len(sample):,} photos in the sample: {len(sample) - len(todo):,} already on disk, "
          f"{len(todo):,} to download")

    # Step 4: Download several photos at the same time (most of the time is spent waiting on the network)
    with ThreadPoolExecutor(max_workers=config["sample"]["download_workers"]) as pool:
        results = list(tqdm(pool.map(fetch_and_resize, todo["url"], todo["image_path"], [size] * len(todo)),
                            total=len(todo)))

    # Step 5: Report and save the photos that could not be downloaded
    failed = todo.assign(reason=results).dropna(subset=["reason"])
    failed.drop(columns="image_path").to_csv(processed / "failed_downloads.csv", index=False)

    print(f"\nDownloaded: {len(todo) - len(failed):,}   Failed: {len(failed):,}")
    if len(failed):
        print(failed["reason"].value_counts().to_string())
        print(f"Failed photos are listed in {processed / 'failed_downloads.csv'}")


if __name__ == "__main__":
    main()
