"""
FastAPI serving app for the food11 model, loaded from the mlflow Model
Registry via its `champion` alias (not from a local .pth file).

Run locally:
    uv run uvicorn src.food11.serve:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST -F "file=@data/food11_processed_mini/validation/Bread/<some-file>.jpg" http://127.0.0.1:8000/predict
"""

import os
from io import BytesIO

import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_URI = "models:/food11@champion"

CATEGORIES = [
    "Bread",
    "Dairy product",
    "Dessert",
    "Egg",
    "Fried food",
    "Meat",
    "Noodles-Pasta",
    "Rice",
    "Seafood",
    "Soup",
    "Vegetable-Fruit",
]

# Same preprocessing used at training time (128x128, ImageNet normalization)
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

app = FastAPI()
model = None


@app.on_event("startup")
def load_model() -> None:
    """Load the model once at process startup, not per-request."""
    global model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    image_bytes = await file.read()
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    tensor = transform(image).unsqueeze(0).numpy()  # shape (1, 3, 128, 128)

    logits = np.array(model.predict(tensor))
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    predicted_idx = int(np.argmax(probs, axis=1)[0])
    confidence = float(probs[0][predicted_idx])

    return {
        "category": CATEGORIES[predicted_idx],
        "confidence": confidence,
    }
