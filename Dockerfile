# Docker image for the API
#   docker build -t camtrap-api .
#   docker run -p 8000:8000 camtrap-api
# then open http://localhost:8000/docs
#
# The trained model (models/) is copied into the image, so the image runs anywhere on its own.

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Step 1: Install the packages first, so Docker can reuse this layer when only the code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Step 2: The code, the settings and the trained model
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

# Step 3: Start the API
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "camtrap.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
