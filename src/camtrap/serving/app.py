# =====================================================================
# API service: send a camera-trap photo, get the species back
# =====================================================================
# Endpoints
#   GET  /health    is the service up, and is a model loaded?
#   POST /predict   upload one photo (JPEG or PNG). Returns the species,
#                   the confidence, the top 3 answers and whether a person
#                   should check the photo.
#
# The model is loaded once, when the service starts, and only if it passed
# the quality gate in training (validation macro-F1 >= 0.50).
# The calibration and the review threshold come from evaluate.py
# (models/serving.json), so the API flags photos the same way as the report.
#
# How to run it (from the project folder):
#   uvicorn camtrap.serving.app:app
# then open http://127.0.0.1:8000/docs to try it in the browser.
# =====================================================================

import io
import json
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from camtrap.data.download import load_config

# Everything the service needs; filled in when it starts
service = {"model": None, "model_name": None, "class_names": [], "image_size": 224,
           "threshold": None, "calibration": None}


# Step 1: Load the final model and its settings
def load_service(config):
    models_dir = Path(config["paths"]["models"])
    settings_file = models_dir / "serving.json"
    if not settings_file.exists():
        print("models/serving.json not found: run `python -m camtrap.evaluation.evaluate` first")
        return

    settings = json.loads(settings_file.read_text())
    name = settings["model"]
    info = json.loads((models_dir / f"{name}.json").read_text())
    if not info["passed_quality_gate"]:
        print(f"{name} did not pass the quality gate, so no model is loaded")
        return

    # TensorFlow is imported here, not at the top, so the tests can run without it
    from tensorflow import keras
    service.update(model=keras.models.load_model(models_dir / f"{name}.keras"),
                   model_name=name,
                   class_names=info["class_names"],
                   image_size=info["image_size"],
                   threshold=settings["review_threshold"],
                   calibration=settings["calibration"])
    print(f"Loaded {name} ({len(info['class_names'])} classes)")


@asynccontextmanager
async def lifespan(app):
    load_service(load_config())
    yield


app = FastAPI(title="Camera-trap species classifier",
              description="Upload a Snapshot Serengeti-style photo; get the species and a review flag.",
              lifespan=lifespan)


# Step 2: What the API sends back (FastAPI checks the answer against these)
class Answer(BaseModel):
    species: str
    probability: float


class Prediction(BaseModel):
    species: str                    # "empty" if no animal is seen
    confidence: float               # the model's own probability for that species
    calibrated_confidence: float    # how often answers this confident were right on validation cameras
    needs_review: bool              # True if a person should check the photo
    review_threshold: float
    top_3: list[Answer]
    model: str


# Step 3: Prepare a photo the same way as in training (RGB, 224 x 224, values 0-255)
def prepare(image, size):
    image = image.convert("RGB").resize((size, size))
    return np.asarray(image, dtype="float32")[None, ...]   # add the batch dimension


def predict_image(image):
    probs = service["model"].predict(prepare(image, service["image_size"]), verbose=0)[0]
    order = np.argsort(probs)[::-1]
    names = service["class_names"]

    confidence = float(probs[order[0]])
    cal = service["calibration"]
    calibrated = float(np.interp(confidence, cal["confidence"], cal["calibrated"]))

    return Prediction(species=names[order[0]],
                      confidence=round(confidence, 4),
                      calibrated_confidence=round(calibrated, 4),
                      needs_review=calibrated < service["threshold"],
                      review_threshold=service["threshold"],
                      top_3=[Answer(species=names[i], probability=round(float(probs[i]), 4)) for i in order[:3]],
                      model=service["model_name"])


# Step 4: The endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": service["model"] is not None, "model": service["model_name"]}


@app.post("/predict", response_model=Prediction)
async def predict(file: UploadFile = File(...)):
    if service["model"] is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The file is not a readable image")
    return predict_image(image)
