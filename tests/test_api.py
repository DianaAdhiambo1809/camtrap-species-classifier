# Tests for the API. A small fake model stands in for the CNN,
# so these run in seconds and without TensorFlow (also on GitHub Actions).
#   pytest

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from camtrap.serving.app import app, service

CLASSES = ["empty", "giraffe", "lion", "zebra"]


class FakeModel:
    def __init__(self, probs):
        self.probs = np.array([probs], dtype="float32")

    def predict(self, x, verbose=0):
        assert x.shape == (1, 224, 224, 3)    # the photo reaches the model at the training size
        return self.probs


def use_model(probs):
    service.update(model=FakeModel(probs), model_name="fake", class_names=CLASSES, image_size=224,
                   threshold=0.6, calibration={"confidence": [0.0, 1.0], "calibrated": [0.0, 1.0]})


def photo_bytes(size=(640, 480), fmt="JPEG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 100, 60)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def reset_service():
    saved = dict(service)
    yield
    service.clear()
    service.update(saved)


client = TestClient(app)   # not used as a context manager, so the real model is never loaded


def test_health():
    use_model([0.1, 0.7, 0.1, 0.1])
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model": "fake"}


def test_confident_prediction():
    use_model([0.05, 0.85, 0.06, 0.04])
    response = client.post("/predict", files={"file": ("giraffe.jpg", photo_bytes(), "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == "giraffe"
    assert body["needs_review"] is False
    assert [a["species"] for a in body["top_3"]] == ["giraffe", "lion", "empty"]


def test_unsure_prediction_goes_to_review():
    use_model([0.3, 0.2, 0.4, 0.1])
    body = client.post("/predict", files={"file": ("x.jpg", photo_bytes(), "image/jpeg")}).json()
    assert body["species"] == "lion"
    assert body["needs_review"] is True


def test_png_and_odd_sizes_work():
    use_model([0.9, 0.05, 0.03, 0.02])
    response = client.post("/predict", files={"file": ("x.png", photo_bytes((100, 300), "PNG"), "image/png")})
    assert response.status_code == 200
    assert response.json()["species"] == "empty"


def test_not_an_image():
    use_model([0.25, 0.25, 0.25, 0.25])
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_no_model_loaded():
    service.update(model=None, model_name=None)
    assert client.get("/health").json()["model_loaded"] is False
    response = client.post("/predict", files={"file": ("x.jpg", photo_bytes(), "image/jpeg")})
    assert response.status_code == 503
