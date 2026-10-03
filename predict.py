import os
import sys
import random
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH  = Path("cifake_model.keras")
CLASS_NAMES = ["REAL", "AI-Generated"]
IMAGE_SIZE  = (32, 32)


def load_image(img_path: Path) -> np.ndarray:
    img   = Image.open(img_path).convert("RGB")
    img   = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def predict_image(model: tf.keras.Model, img_path: Path) -> dict:
    tensor     = load_image(img_path)
    prob_ai    = float(model.predict(tensor, verbose=0)[0][0])
    pred_index = 1 if prob_ai >= 0.5 else 0
    pred_label = CLASS_NAMES[pred_index]
    confidence = prob_ai if pred_index == 1 else (1.0 - prob_ai)
    return {
        "path":       str(img_path),
        "prob_ai":    prob_ai,
        "pred_label": pred_label,
        "confidence": confidence * 100.0,
    }


def main() -> None:
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

    if len(sys.argv) > 1:
        image_entries = [(Path(p), "Unknown") for p in sys.argv[1:]]
    else:
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

    col_path   = 45
    col_actual = 12
    col_pred   = 14
    col_prob   = 8

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

        result     = predict_image(model, img_path)
        is_correct = actual_label == "Unknown" or result["pred_label"] == actual_label
        status     = "✓" if is_correct else "✗"

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
