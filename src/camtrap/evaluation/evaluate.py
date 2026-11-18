# =====================================================================
# Evaluate the models on the test cameras
# =====================================================================
# Works from the prediction files made by predict.py and baselines.py, so it
# runs on any computer without TensorFlow.
#
# For every model: accuracy, macro-F1 and Brier score, on validation and test.
# The final model is the one with the best VALIDATION macro-F1, chosen before
# looking at the test results. For that model, on the test cameras:
#   - precision and recall per class, confusion matrix
#   - macro-F1 with a 95% interval (bootstrap over test cameras)
#   - day vs night
#   - the empty filter: how many animals would be thrown away
#   - calibration, and the confidence threshold for human review chosen by cost
#
# How to run it (from the project folder):
#   python -m camtrap.evaluation.evaluate
#
# Needs:  reports/results/predictions_*.csv, data/processed/sample.csv
# Output: reports/results/evaluation_summary.csv, test_report.csv, test_details.json
#         reports/figures/test_confusion_matrix.png, test_calibration.png, test_review_curve.png
# =====================================================================

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, average_precision_score,
                             classification_report, f1_score)

from camtrap.data.download import load_config


# Step 1: Read one model's predictions; add the camera and time of day of each photo
def load_predictions(path, sample):
    pred = pd.read_csv(path)
    info = sample.assign(file=sample["photo"].apply(lambda p: Path(p).stem + ".jpg"))
    pred = pred.merge(info[["file", "location", "time_of_day"]], on="file", how="left")

    prob_cols = [c for c in pred.columns if c.startswith("prob_")]
    class_names = [c.removeprefix("prob_") for c in prob_cols]
    pred["confidence"] = pred[prob_cols].max(axis=1)
    pred["correct"] = pred["true"] == pred["pred"]
    return pred, class_names, prob_cols


# Multi-class Brier score: average squared gap between the probabilities and the true answer.
# 0 is perfect; lower is better. (Lab 3 used the two-class version.)
def brier(pred, class_names, prob_cols):
    truth = (pred["true"].to_numpy()[:, None] == np.array(class_names)[None, :]).astype(float)
    return np.mean(np.sum((pred[prob_cols].to_numpy() - truth) ** 2, axis=1))


def scores(pred, class_names, prob_cols):
    return {"accuracy": accuracy_score(pred["true"], pred["pred"]),
            "macro_f1": f1_score(pred["true"], pred["pred"], average="macro"),
            "brier": brier(pred, class_names, prob_cols),
            "photos": len(pred)}


# Step 2: 95% interval for macro-F1. Cameras, not single photos, are resampled,
#         because photos from the same camera are not independent.
def bootstrap_macro_f1(test, rounds, seed):
    rng = np.random.default_rng(seed)
    groups = {camera: g for camera, g in test.groupby("location")}
    cameras = list(groups)

    results = []
    for _ in range(rounds):
        chosen = rng.choice(cameras, size=len(cameras), replace=True)
        resample = pd.concat([groups[c] for c in chosen])
        results.append(f1_score(resample["true"], resample["pred"], average="macro"))
    low, high = np.percentile(results, [2.5, 97.5])
    return float(low), float(high)


# Step 3: The empty filter: a photo is thrown away if the model says "empty"
def empty_filter(test):
    is_animal = test["true"] != "empty"
    kept = test["pred"] != "empty"
    return {
        "animals_kept": float((kept & is_animal).sum() / is_animal.sum()),       # recall of animals
        "empty_removed": float((~kept & ~is_animal).sum() / (~is_animal).sum()),
        "pr_auc_animal": float(average_precision_score(is_animal, 1 - test["prob_empty"])),
    }


# Step 4: Review threshold. Photos below the threshold go to a person (cost: review_cost each);
#         photos above it are accepted, and each wrong one costs error_cost.
#         The threshold with the lowest total cost on VALIDATION is kept (as in Lab 3).
def total_cost(confidence, correct, threshold, review_cost, error_cost):
    review = confidence < threshold
    return review.sum() * review_cost + (~review & ~correct).sum() * error_cost


def choose_threshold(val, review_cost, error_cost):
    candidates = np.unique(np.r_[0.0, val["calibrated"], 1.01])   # 0 = review nothing, 1.01 = review all
    costs = [total_cost(val["calibrated"], val["correct"], t, review_cost, error_cost) for t in candidates]
    return float(candidates[int(np.argmin(costs))])


def plot_review_curve(test, threshold, path):
    ordered = test.sort_values("calibrated", ascending=False)
    share_accepted = np.arange(1, len(ordered) + 1) / len(ordered)
    accuracy_accepted = ordered["correct"].cumsum().to_numpy() / np.arange(1, len(ordered) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(share_accepted, accuracy_accepted, color="steelblue")
    chosen = (test["calibrated"] >= threshold).mean()
    plt.axvline(chosen, color="red", linestyle="--", label=f"chosen threshold: {chosen:.0%} accepted")
    plt.xlabel("Share of photos accepted automatically")
    plt.ylabel("Accuracy of the accepted photos")
    plt.title("Review curve, test cameras\n(the rest go to a person)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_calibration(test, path):
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
    for column, name in [("confidence", "model confidence"), ("calibrated", "after isotonic calibration")]:
        observed, predicted = calibration_curve(test["correct"], test[column], n_bins=10, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=name)
    plt.xlabel("Confidence")
    plt.ylabel("Share actually correct")
    plt.title("Calibration, test cameras")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    config = load_config()
    ev = config["evaluation"]
    processed = Path(config["paths"]["processed"])
    results = Path(config["paths"]["reports"]) / "results"
    figures = Path(config["paths"]["reports"]) / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    files = sorted(results.glob("predictions_*.csv"))
    if not files:
        print("No prediction files yet: run baselines.py, and predict.py in Colab, first.")
        return
    sample = pd.read_csv(processed / "sample.csv")

    # Scores of every model on validation and test
    rows, predictions = [], {}
    for path in files:
        name = path.stem.removeprefix("predictions_")
        pred, class_names, prob_cols = load_predictions(path, sample)
        predictions[name] = (pred, class_names, prob_cols)
        for split in ["val", "test"]:
            rows.append({"model": name, "split": split,
                         **scores(pred[pred["split"] == split], class_names, prob_cols)})
    summary = pd.DataFrame(rows)
    summary.to_csv(results / "evaluation_summary.csv", index=False)
    print(summary.pivot(index="model", columns="split", values=["macro_f1", "accuracy", "brier"]).round(3))

    # The final model is picked on validation only
    best = summary[summary["split"] == "val"].sort_values("macro_f1").iloc[-1]["model"]
    pred, class_names, prob_cols = predictions[best]
    val, test = pred[pred["split"] == "val"].copy(), pred[pred["split"] == "test"].copy()
    print(f"\nFinal model (best validation macro-F1): {best}")

    # Per-class results and confusion matrix
    report = classification_report(test["true"], test["pred"], output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(3).to_csv(results / "test_report.csv")
    print(classification_report(test["true"], test["pred"], digits=3, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(test["true"], test["pred"], labels=class_names, normalize="true",
                                            values_format=".2f", xticks_rotation=90, cmap="Blues")
    plt.title(f"{best}, test cameras (rows = true class, share of each row)")
    plt.gcf().set_size_inches(10, 9)
    plt.tight_layout()
    plt.savefig(figures / "test_confusion_matrix.png", dpi=120)
    plt.close()

    # Calibration: learn on validation how often each confidence level is right, apply to test
    iso = IsotonicRegression(out_of_bounds="clip").fit(val["confidence"], val["correct"])
    val["calibrated"] = iso.predict(val["confidence"])
    test["calibrated"] = iso.predict(test["confidence"])
    plot_calibration(test, figures / "test_calibration.png")

    threshold = choose_threshold(val, ev["review_cost"], ev["error_cost"])
    accepted = test["calibrated"] >= threshold
    plot_review_curve(test, threshold, figures / "test_review_curve.png")

    per_100 = lambda cost: round(100 * cost / len(test), 1)
    low, high = bootstrap_macro_f1(test, ev["bootstrap_rounds"], config["seed"])
    known_time = test[test["time_of_day"] != "unknown"]

    details = {
        "final_model": best,
        "test_macro_f1": round(f1_score(test["true"], test["pred"], average="macro"), 4),
        "test_macro_f1_95ci": [round(low, 4), round(high, 4)],
        "val_minus_test_macro_f1": round(
            f1_score(val["true"], val["pred"], average="macro") - f1_score(test["true"], test["pred"], average="macro"), 4),
        "day_vs_night": {period: {"macro_f1": round(f1_score(g["true"], g["pred"], average="macro"), 4),
                                  "accuracy": round(accuracy_score(g["true"], g["pred"]), 4),
                                  "photos": len(g)}
                         for period, g in known_time.groupby("time_of_day")},
        "empty_filter": empty_filter(test),
        # Brier score of the confidence ("is the top answer right?"), before and after calibration
        "confidence_brier_before_after": [
            round(float(np.mean((test["confidence"] - test["correct"]) ** 2)), 4),
            round(float(np.mean((test["calibrated"] - test["correct"]) ** 2)), 4)],
        "review": {
            "threshold": round(threshold, 4),
            "share_sent_to_review": round(float((~accepted).mean()), 4),
            "accuracy_of_accepted": round(float(test.loc[accepted, "correct"].mean()), 4),
            "cost_per_100_photos": {
                "model_with_review": per_100(total_cost(test["calibrated"], test["correct"], threshold,
                                                        ev["review_cost"], ev["error_cost"])),
                "review_everything": per_100(len(test) * ev["review_cost"]),
                "accept_everything": per_100((~test["correct"]).sum() * ev["error_cost"]),
            },
        },
    }
    (results / "test_details.json").write_text(json.dumps(details, indent=2))
    print(json.dumps(details, indent=2))


if __name__ == "__main__":
    main()
