import os
import io
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

MODEL_PATH = Path(__file__).parent.parent / "cifake_model.keras"

app = FastAPI(title="CIFAKE Detector API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: tf.keras.Model | None = None


@app.on_event("startup")
async def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'.\n"
            "Please run 'python train.py' first to train and save the model."
        )
    print(f"Loading model from '{MODEL_PATH}' ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded.")


class PredictionResponse(BaseModel):
    label:      str
    prob_ai:    float
    prob_real:  float
    confidence: float


@app.get("/")
async def health_check() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPEG, PNG, WEBP, or BMP image.",
        )

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    prob_ai   = float(model.predict(arr, verbose=0)[0][0])
    prob_real = 1.0 - prob_ai
    is_ai     = prob_ai >= 0.5

    return PredictionResponse(
        label      = "AI-Generated" if is_ai else "Real",
        prob_ai    = round(prob_ai, 4),
        prob_real  = round(prob_real, 4),
        confidence = round(prob_ai if is_ai else prob_real, 4),
    )
