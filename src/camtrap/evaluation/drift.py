# =====================================================================
# Drift report: do the test cameras see a different world from the training cameras?
# =====================================================================
# A deployed model meets new cameras. Evidently compares the photos of the
# training cameras (reference) with those of the test cameras (current), using:
#   - simple measurements of each photo: brightness, contrast, colour
#     (night infrared photos are grey, so colour is close to 0), share of green
#   - the hour the photo was taken
#   - the first 5 principal components of the pretrained image features
#     (PCA, as in Lab 5: fitted on the training cameras only)
# The text strip at the bottom of each photo is left out of the measurements.
#
# How to run it (from the project folder):
#   python -m camtrap.evaluation.drift
#
# Needs:  data/images/, data/processed/sample.csv and features.npz
# Output: reports/drift_report.html            (open in a browser)
#         reports/results/drift_summary.csv
#         reports/figures/pca_train_vs_test.png
# =====================================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from camtrap.data.download import load_config


# Step 1: Simple measurements of one photo, without the text strip
def photo_measurements(path, strip_rows):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=float)[:-strip_rows]
    grey = a.mean(axis=2)
    return {"brightness": grey.mean(),
            "contrast": grey.std(),
            "colour": (np.abs(a[..., 0] - a[..., 1]) + np.abs(a[..., 1] - a[..., 2])).mean(),
            "green_share": a[..., 1].sum() / (a.sum() + 1e-8)}


# Step 2: One table per split: measurements, hour and principal components
def build_table(split, features, pca, scaler, sample, images, strip_rows):
    files = features[f"files_{split}"]
    table = pd.DataFrame({"file": files})
    table = table.merge(sample[["file", "location", "class", "start"]], on="file", how="left")
    table["hour"] = pd.to_datetime(table["start"]).dt.hour

    paths = [images / split / c / f for c, f in zip(table["class"], table["file"])]
    table = pd.concat([table, pd.DataFrame([photo_measurements(p, strip_rows) for p in paths])], axis=1)

    components = pca.transform(scaler.transform(features[f"X_{split}"]))
    for i in range(components.shape[1]):
        table[f"pc{i + 1}"] = components[:, i]
    return table


def plot_pca(train, test, path):
    plt.figure(figsize=(7, 6))
    plt.scatter(train["pc1"], train["pc2"], s=4, alpha=0.3, label="training cameras")
    plt.scatter(test["pc1"], test["pc2"], s=4, alpha=0.3, label="test cameras")
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.title("Pretrained image features, training vs test cameras")
    plt.legend(markerscale=4)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    config = load_config()
    processed = Path(config["paths"]["processed"])
    images = Path(config["paths"]["images"])
    reports = Path(config["paths"]["reports"])
    strip_rows = config["gradcam"]["strip_rows"]

    sample = pd.read_csv(processed / "sample.csv")
    sample["file"] = sample["photo"].apply(lambda p: Path(p).stem + ".jpg")
    features = np.load(processed / "features.npz")

    # PCA learnt on the training cameras only
    scaler = StandardScaler().fit(features["X_train"])
    pca = PCA(n_components=5, random_state=config["seed"]).fit(scaler.transform(features["X_train"]))
    print(f"Variance kept by 5 components: {pca.explained_variance_ratio_.sum():.1%}")

    print("Measuring the photos...")
    train = build_table("train", features, pca, scaler, sample, images, strip_rows)
    test = build_table("test", features, pca, scaler, sample, images, strip_rows)
    plot_pca(train, test, reports / "figures" / "pca_train_vs_test.png")

    # Step 3: Evidently data drift report, training cameras (reference) vs test cameras (current)
    columns = ["brightness", "contrast", "colour", "green_share", "hour", "pc1", "pc2", "pc3", "pc4", "pc5"]
    definition = DataDefinition(numerical_columns=columns)
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=Dataset.from_pandas(train[columns], data_definition=definition),
                        current_data=Dataset.from_pandas(test[columns], data_definition=definition))
    result.save_html(str(reports / "drift_report.html"))

    # Step 4: A small table for the report: the average of each column and Evidently's drift test.
    #         Evidently picks the test by sample size: for a p-value, drift means below the threshold;
    #         for a distance, drift means at or above it.
    rows = {}
    for metric in result.dict()["metrics"]:
        settings = metric["config"]
        if settings["type"].endswith("ValueDrift"):
            score, threshold = metric["value"], settings["threshold"]
            drifted = score < threshold if "p_value" in settings["method"] else score >= threshold
            rows[settings["column"]] = {"method": settings["method"], "threshold": settings["threshold"],
                                        "drift_score": metric["value"], "drift_detected": drifted}
    summary = pd.concat([train[columns].mean().rename("train_mean"), test[columns].mean().rename("test_mean"),
                         pd.DataFrame(rows).T], axis=1)
    summary.round(4).to_csv(reports / "results" / "drift_summary.csv", index_label="column")
    print(summary.round(3))
    print("\nOpen reports/drift_report.html in a browser for the full report.")


if __name__ == "__main__":
    main()
