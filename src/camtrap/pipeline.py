# =====================================================================
# Run the whole pipeline with one command
# =====================================================================
# Runs every step in order: data -> features -> training -> prediction ->
# baselines -> evaluation -> drift report -> Grad-CAM.
#
# A step is skipped when its output already exists, so the pipeline can be
# stopped and started again. Use --force to redo every step.
# Steps that need TensorFlow (features, training, prediction, Grad-CAM) are
# skipped with a message if TensorFlow is not installed; run them on Colab
# (notebooks/colab_run.ipynb) or on a computer with a GPU.
#
# How to run it (from the project folder):
#   python -m camtrap.pipeline           run what is missing
#   python -m camtrap.pipeline --force   run everything again
# =====================================================================

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

# (what the step does, command, output that shows it is done, needs TensorFlow)
STEPS = [
    ("Download the labels", "camtrap.data.download", "data/raw/SnapshotSerengetiS01.json", False),
    ("Clean", "camtrap.data.clean", "data/processed/bursts.csv", False),
    ("Split by camera and pick the sample", "camtrap.data.split", "data/processed/sample.csv", False),
    ("Download the photos", "camtrap.data.download_images", None, False),   # skips photos already on disk
    ("Pretrained image features", "camtrap.models.features", "data/processed/features.npz", True),
    ("Train the CNN from scratch", "camtrap.models.train --model scratch", "models/scratch.json", True),
    ("Train the pretrained CNN", "camtrap.models.train --model pretrained", "models/pretrained.json", True),
    ("Predict validation and test photos", "camtrap.models.predict",
     "reports/results/predictions_pretrained.csv", True),
    ("Baselines", "camtrap.models.baselines", "reports/results/baselines.csv", False),
    ("Evaluate on the test cameras", "camtrap.evaluation.evaluate", "reports/results/test_details.json", False),
    ("Drift report", "camtrap.evaluation.drift", "reports/drift_report.html", False),
    ("Grad-CAM and date-strip check", "camtrap.evaluation.gradcam", "reports/results/gradcam_strip_check.json", True),
]


def main():
    parser = argparse.ArgumentParser(description="Run the whole pipeline")
    parser.add_argument("--force", action="store_true", help="run every step, even if its output exists")
    args = parser.parse_args()

    has_tensorflow = importlib.util.find_spec("tensorflow") is not None

    for number, (name, command, output, needs_tf) in enumerate(STEPS, start=1):
        title = f"Step {number}/{len(STEPS)}: {name}"
        if output and Path(output).exists() and not args.force:
            print(f"{title}: already done ({output})", flush=True)
            continue
        if needs_tf and not has_tensorflow:
            print(f"{title}: skipped, needs TensorFlow (run it on Colab)", flush=True)
            continue

        print(f"\n{title}\n  python -m {command}", flush=True)
        result = subprocess.run([sys.executable, "-m", *command.split()])
        if result.returncode != 0:
            sys.exit(f"{title} failed; fix it and run the pipeline again (finished steps are skipped)")

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
