import os
import sqlite3
import numpy as np
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

BASE_DIR = Path(__file__).parent
DB_PATH  = BASE_DIR / "database.db"

MODEL_PATH = BASE_DIR / "cnn_model.keras"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR / "cnn_model.h5"

print(f"Loading model from '{MODEL_PATH}' ...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found at '{MODEL_PATH}'.\n"
        "Copy your trained model into this folder:\n"
        "  cp cifake_model.keras ai-detector-app/backend/cnn_model.keras"
    )

model = tf.keras.models.load_model(str(MODEL_PATH))
print("✓ Model loaded successfully.")


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH), detect_types=sqlite3.PARSE_DECLTYPES)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    with app.app_context():
        db = get_db()
        db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER  PRIMARY KEY AUTOINCREMENT,
                filename         TEXT     NOT NULL,
                is_ai            BOOLEAN  NOT NULL,
                confidence_score FLOAT    NOT NULL,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        db.commit()


def preprocess_image(file_stream) -> np.ndarray:
    img = Image.open(file_stream).convert("RGB")
    img = img.resize((32, 32), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image file uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "Empty filename."}), 400

    filename = file.filename

    tensor     = preprocess_image(file.stream)
    raw_output = float(model.predict(tensor, verbose=0)[0][0])

    is_ai      = raw_output >= 0.5
    confidence = raw_output if is_ai else (1.0 - raw_output)
    label      = "AI-Generated" if is_ai else "Real Image"

    db = get_db()
    db.execute(
        "INSERT INTO predictions (filename, is_ai, confidence_score) VALUES (?, ?, ?)",
        (filename, bool(is_ai), float(raw_output)),
    )
    db.commit()

    return jsonify({
        "status":           "success",
        "filename":         filename,
        "prediction":       label,
        "confidence_score": round(raw_output, 4),
        "is_ai":            bool(is_ai),
    })


@app.route("/api/history", methods=["GET"])
def history():
    db   = get_db()
    rows = db.execute(
        "SELECT id, filename, is_ai, confidence_score, timestamp "
        "FROM predictions ORDER BY timestamp DESC LIMIT 10"
    ).fetchall()

    return jsonify([
        {
            "id":               row["id"],
            "filename":         row["filename"],
            "is_ai":            bool(row["is_ai"]),
            "confidence_score": row["confidence_score"],
            "timestamp":        row["timestamp"],
        }
        for row in rows
    ])


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
