"""
backend/main.py
===============
FastAPI server that loads the trained CIFAKE model and serves predictions.

Endpoints:
  GET  /            → Health check
  POST /predict     → Upload an image, receive real/AI prediction + probability
"""

import os
import io
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load model once at startup ─────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent / "cifake_model.keras"

app = FastAPI(title="CIFAKE Detector API", version="1.0.0")

# Allow Next.js dev server (port 3000) and production origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: tf.keras.Model | None = None


@app.on_event("startup")
async def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at '{MODEL_PATH}'.\n"
            "Please run 'python train.py' first to train and save the model."
        )
    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded.")


class PredictionResponse(BaseModel):
    label: str           # "Real" or "AI-Generated"
    prob_ai: float       # probability of being AI-generated  (0.0 – 1.0)
    prob_real: float     # probability of being real          (0.0 – 1.0)
    confidence: float    # confidence of the winning class    (0.0 – 1.0)


@app.get("/")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload a JPEG, PNG, WEBP or BMP image.",
        )

    # Read and preprocess image
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = img.resize((32, 32), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, 32, 32, 3)

    # Inference
    prob_ai = float(model.predict(arr, verbose=0)[0][0])
    prob_real = 1.0 - prob_ai
    is_ai = prob_ai >= 0.5

    return PredictionResponse(
        label="AI-Generated" if is_ai else "Real",
        prob_ai=round(prob_ai, 4),
        prob_real=round(prob_real, 4),
        confidence=round(prob_ai if is_ai else prob_real, 4),
    )
