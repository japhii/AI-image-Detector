"""
predict.py
==========
Command-line tool for running predictions with the trained CNN model.

Usage
-----
  # Predict a single image:
  python predict.py data/test/FAKE/1000.jpg

  # Predict multiple images at once:
  python predict.py data/test/REAL/10.jpg data/test/FAKE/20.jpg

  # No arguments: randomly sample 5 real + 5 fake images from data/test/
  python predict.py
"""

import os
import sys
import random
from pathlib import Path

# Silence TensorFlow startup logs (show only errors)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

import numpy as np
import tensorflow as tf
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = Path("cifake_model.keras")
CLASS_NAMES = ["REAL", "AI-Generated"]   # index 0 = real, index 1 = AI-generated
IMAGE_SIZE  = (32, 32)


# ── Image helpers ─────────────────────────────────────────────────────────────

def load_image(img_path: Path) -> np.ndarray:
    """
    Load an image from disk, resize to 32×32 RGB, and scale pixels to [0, 1].
    Returns a numpy array of shape (1, 32, 32, 3) ready for model input.
    """
    img   = Image.open(img_path).convert("RGB")
    img   = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)   # add batch dimension → (1, 32, 32, 3)


def predict_image(model: tf.keras.Model, img_path: Path) -> dict:
    """
    Run the model on a single image and return a dictionary with:
      - path         : the file path
      - prob_ai      : raw sigmoid output (0 = real, 1 = AI-generated)
      - pred_label   : human-readable label ("REAL" or "AI-Generated")
      - confidence   : confidence percentage for the predicted label
    """
    tensor   = load_image(img_path)
    prob_ai  = float(model.predict(tensor, verbose=0)[0][0])

    pred_index = 1 if prob_ai >= 0.5 else 0
    pred_label = CLASS_NAMES[pred_index]
    confidence = prob_ai if pred_index == 1 else (1.0 - prob_ai)

    return {
        "path":        str(img_path),
        "prob_ai":     prob_ai,
        "pred_label":  pred_label,
        "confidence":  confidence * 100.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # Check the model file exists before doing anything else
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model file '{MODEL_PATH}' was not found.")
        print("Please run 'python train.py' first to train and save the model.\n")
        sys.exit(1)

    print("=" * 70)
    print("  Real vs AI-Generated Image Classifier — Predictor")
    print("=" * 70)
    print(f"  Loading model from '{MODEL_PATH}' ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("  Model loaded successfully.\n")

    # ── Decide which images to test ───────────────────────────────────────────
    if len(sys.argv) > 1:
        # User provided specific image paths as command-line arguments
        image_entries = [(Path(p), "Unknown") for p in sys.argv[1:]]

    else:
        # Default: pick 5 random REAL + 5 random FAKE from data/test/
        print("No image path given. Sampling 10 random images from data/test/ ...\n")
        test_dir = Path("data/test")

        if not test_dir.exists():
            print(f"[ERROR] '{test_dir}' folder not found. Run setup_data.py first.")
            sys.exit(1)

        real_imgs = list((test_dir / "REAL").glob("*.jpg")) + \
                    list((test_dir / "REAL").glob("*.png"))
        fake_imgs = list((test_dir / "FAKE").glob("*.jpg")) + \
                    list((test_dir / "FAKE").glob("*.png"))

        sampled_real = random.sample(real_imgs, min(5, len(real_imgs)))
        sampled_fake = random.sample(fake_imgs, min(5, len(fake_imgs)))

        image_entries = (
            [(p, "REAL")         for p in sampled_real] +
            [(p, "AI-Generated") for p in sampled_fake]
        )

    # ── Print results table ───────────────────────────────────────────────────
    col_path   = 45
    col_actual = 12
    col_pred   = 14
    col_prob   = 8
    col_conf   = 10

    header = (
        f"{'Image Path':<{col_path}} | "
        f"{'Actual':<{col_actual}} | "
        f"{'Predicted':<{col_pred}} | "
        f"{'P(AI)':<{col_prob}} | "
        f"Confidence"
    )
    print(header)
    print("-" * len(header))

    for img_path, actual_label in image_entries:
        if not img_path.exists():
            print(f"{str(img_path):<{col_path}} | File not found!")
            continue

        result = predict_image(model, img_path)
        is_correct = actual_label == "Unknown" or result["pred_label"] == actual_label
        status = "✓" if is_correct else "✗"

        print(
            f"{result['path']:<{col_path}} | "
            f"{actual_label:<{col_actual}} | "
            f"{result['pred_label']:<{col_pred}} | "
            f"{result['prob_ai']:.4f}   | "
            f"{result['confidence']:.1f}% {status}"
        )

    print("-" * len(header))
    print("\nP(AI) scale: 0.0 = Definitely Real  →  1.0 = Definitely AI-Generated")
    print("Decision threshold: P(AI) >= 0.5 → classified as AI-Generated\n")


if __name__ == "__main__":
    main()
