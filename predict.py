"""
predict.py
==========
Inference & testing script for the trained Real vs AI-Generated Image Classifier.

Usage Options:
  1. Test a single image:
     python predict.py data/test/FAKE/1000.jpg

  2. Test multiple images:
     python predict.py data/test/REAL/10.jpg data/test/FAKE/20.jpg

  3. Test on 10 random samples from the dataset (no args passed):
     python predict.py
"""

import sys
import os
import random
from pathlib import Path

# Silence TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("cifake_model.keras")
CLASS_NAMES = ["REAL", "AI-Generated"]  # 0 = REAL, 1 = AI-Generated


def load_and_preprocess_image(img_path: Path) -> np.ndarray:
    """Load an image file, resize to 32x32 RGB, and scale pixels to [0, 1]."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((32, 32), Image.Resampling.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Add batch dimension -> shape (1, 32, 32, 3)
    return np.expand_dims(img_array, axis=0)


def predict_image(model: tf.keras.Model, img_path: Path) -> dict:
    """Run prediction on a single image path."""
    x = load_and_preprocess_image(img_path)
    prob_ai = float(model.predict(x, verbose=0)[0][0])
    
    pred_class = 1 if prob_ai >= 0.5 else 0
    pred_label = CLASS_NAMES[pred_class]
    confidence = prob_ai if pred_class == 1 else (1.0 - prob_ai)

    return {
        "path": str(img_path),
        "prob_ai": prob_ai,
        "pred_label": pred_label,
        "confidence": confidence * 100.0,
    }


def main():
    if not MODEL_PATH.exists():
        print(f"\n[ERROR] Model file '{MODEL_PATH}' not found.")
        print("Please run 'python train.py' first to train and save the model.\n")
        sys.exit(1)

    print("=" * 70)
    print("  Real vs AI-Generated Image Classifier — Predictor")
    print("=" * 70)
    print(f"Loading trained model from '{MODEL_PATH}' ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.\n")

    # Determine input images to test
    if len(sys.argv) > 1:
        # User provided file paths as arguments
        image_paths = [Path(p) for p in sys.argv[1:]]
        eval_actual = False
    else:
        # Default: Pick 5 random REAL and 5 random FAKE from data/test/
        print("No image path specified. Sampling 10 random images from data/test/ ...\n")
        test_dir = Path("data/test")
        if not test_dir.exists():
            print(f"[ERROR] '{test_dir}' directory not found.")
            sys.exit(1)

        real_imgs = list((test_dir / "REAL").glob("*.jpg")) + list((test_dir / "REAL").glob("*.png"))
        fake_imgs = list((test_dir / "FAKE").glob("*.jpg")) + list((test_dir / "FAKE").glob("*.png"))

        sampled_real = random.sample(real_imgs, min(5, len(real_imgs)))
        sampled_fake = random.sample(fake_imgs, min(5, len(fake_imgs)))
        
        image_paths = [(p, "REAL") for p in sampled_real] + [(p, "AI-Generated") for p in sampled_fake]
        eval_actual = True

    print(f"{'Image Path':<45} | {'Actual':<12} | {'Predicted':<14} | {'P(AI)':<8} | {'Confidence':<10}")
    print("-" * 100)

    for item in image_paths:
        if eval_actual:
            img_path, actual_label = item
        else:
            img_path, actual_label = item, "Unknown"

        if not img_path.exists():
            print(f"{str(img_path):<45} | File Not Found!")
            continue

        res = predict_image(model, img_path)
        
        # Color indicator symbol
        status = "✓" if actual_label == "Unknown" or res["pred_label"] == actual_label else "✗"
        
        print(f"{res['path']:<45} | {actual_label:<12} | {res['pred_label']:<14} | {res['prob_ai']:.4f}   | {res['confidence']:.1f}% {status}")

    print("-" * 100)
    print("\nP(AI) range: 0.0 (Definitely Real) ---> 1.0 (Definitely AI-Generated)")
    print("Decision Threshold: P(AI) >= 0.5 -> Classified as AI-Generated\n")


if __name__ == "__main__":
    main()
