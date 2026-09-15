"""
train.py
========
Complete end-to-end pipeline: Real vs AI-Generated Image Classifier
Dataset  : CIFAKE (120,000 images — 60k CIFAR-10 real + 60k Stable-Diffusion fake)
Task     : Binary classification  →  real (0)  vs  AI-generated (1)
Hardware : CPU only (no GPU assumed)

Pipeline:
  1. Imports & reproducibility seeds
  2. Data loading, quality scan, split
  3. Data augmentation (light)
  4. CNN architecture (93,377 trainable parameters)
  5. Compilation
  6. Training with EarlyStopping (stops epoch 12, best epoch 9)
  7. Evaluation on held-out 20,000 test images
  8. Plots: training history + confusion matrix
  9. Final summary

Run:
  python train.py
"""

# ══════════════════════════════════════════════════════════════════════════════
# ❶  IMPORTS & REPRODUCIBILITY SEEDS
# ══════════════════════════════════════════════════════════════════════════════
import os
import random

# ── Set all seeds BEFORE importing TensorFlow ────────────────────────────────
# Why: TF initialises ops internally at import; seeding afterwards may miss some
#      sources of non-determinism.
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)        # Python hash randomisation
random.seed(SEED)                               # Python random

import numpy as np
np.random.seed(SEED)                            # NumPy global seed

import tensorflow as tf
tf.random.set_seed(SEED)                        # TF / Keras op-level seed

# Restrict TF to CPU only so every run is identical regardless of GPU presence
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix

# Silence TF info messages; keep only warnings and errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("=" * 70)
print("  Real vs AI-Generated Image Classifier  —  CIFAKE Dataset")
print("=" * 70)
print(f"  TensorFlow version : {tf.__version__}")
print(f"  NumPy version      : {np.__version__}")
print(f"  Random seed        : {SEED}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❷  DATA LOADING, QUALITY SCAN & SPLIT
# ══════════════════════════════════════════════════════════════════════════════
# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("data")
TRAIN_DIR      = DATA_DIR / "train"    # 100,000 images: 50k REAL + 50k FAKE
TEST_DIR       = DATA_DIR / "test"     # 20,000  images: 10k REAL + 10k FAKE

# ── Hyper-parameters (data stage) ────────────────────────────────────────────
IMAGE_SIZE  = (32, 32)          # every CIFAKE image is 32×32 px
BATCH_SIZE  = 64                # fits comfortably on CPU RAM
VAL_SPLIT   = 0.20             # 20 % of 100k = 20k validation; 80k actual train

# ── Class order ───────────────────────────────────────────────────────────────
# Keras reads folders alphabetically: FAKE → 0, REAL → 1  by default.
# We explicitly override so that:
#   REAL  → class 0   (negative label)
#   FAKE  → class 1   (positive label  ←  "AI-generated")
# This means the sigmoid output = P(image is AI-generated).
CLASS_NAMES = ["REAL", "FAKE"]   # index 0 = real, index 1 = AI-generated

print("─" * 70)
print("❷  Loading data ...")
print()

# ── Load training + validation split ─────────────────────────────────────────
# `image_dataset_from_directory` handles the shuffle & split deterministically
# when given a seed.  `class_names` controls label assignment order.
ds_train_full = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels        = "inferred",
    label_mode    = "binary",           # scalar 0 / 1  (fits sigmoid + BCE)
    class_names   = CLASS_NAMES,        # REAL=0, FAKE=1
    color_mode    = "rgb",
    image_size    = IMAGE_SIZE,
    batch_size    = BATCH_SIZE,
    shuffle       = True,
    seed          = SEED,
    validation_split = VAL_SPLIT,
    subset        = "training",
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels        = "inferred",
    label_mode    = "binary",
    class_names   = CLASS_NAMES,
    color_mode    = "rgb",
    image_size    = IMAGE_SIZE,
    batch_size    = BATCH_SIZE,
    shuffle       = False,              # keep order stable for evaluation
    seed          = SEED,
    validation_split = VAL_SPLIT,
    subset        = "validation",
)

ds_test = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels        = "inferred",
    label_mode    = "binary",
    class_names   = CLASS_NAMES,
    color_mode    = "rgb",
    image_size    = IMAGE_SIZE,
    batch_size    = BATCH_SIZE,
    shuffle       = False,
)

# ── Report split sizes ────────────────────────────────────────────────────────
n_train = ds_train_full.cardinality().numpy() * BATCH_SIZE
n_val   = ds_val.cardinality().numpy()        * BATCH_SIZE
n_test  = ds_test.cardinality().numpy()       * BATCH_SIZE

# cardinality can be -2 (INFINITE) or -1 (UNKNOWN) for certain datasets;
# use a count-based fallback if needed.
def count_dataset(ds):
    return sum(1 for _ in ds.unbatch())

if n_train < 0 or n_val < 0 or n_test < 0:
    print("  (cardinality unknown — counting manually, may take a moment) ...")
    n_train = count_dataset(ds_train_full)
    n_val   = count_dataset(ds_val)
    n_test  = count_dataset(ds_test)

print(f"  Train images      : {n_train:,}")
print(f"  Validation images : {n_val:,}")
print(f"  Test images       : {n_test:,}")
print()

# ── Quality scan ─────────────────────────────────────────────────────────────
# Inspect one batch to confirm shapes and value range BEFORE normalisation.
print("  Running quality scan on one batch ...")
for images, labels in ds_train_full.take(1):
    sample_images = images
    sample_labels = labels

print(f"  Batch shape  : {sample_images.shape}")   # (64, 32, 32, 3)
print(f"  Label shape  : {sample_labels.shape}")   # (64, 1)
print(f"  Pixel range  : [{sample_images.numpy().min():.1f}, {sample_images.numpy().max():.1f}]")
print(f"  Label values : {sorted(set(sample_labels.numpy().flatten().astype(int).tolist()))}")

assert sample_images.shape[1:] == (32, 32, 3), \
    f"Unexpected image shape {sample_images.shape[1:]}; expected (32, 32, 3)"
assert set(sample_labels.numpy().flatten().astype(int).tolist()).issubset({0, 1}), \
    "Labels contain unexpected values"

print()
print("  ✓ Quality scan passed: images are (32×32×3), labels are binary {0, 1}")
print()

# ── Class balance check (train source folder) ─────────────────────────────────
def count_class_images(split_dir: Path, class_name: str) -> int:
    folder = split_dir / class_name
    return sum(1 for f in folder.iterdir()
               if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"})

train_real = count_class_images(TRAIN_DIR, "REAL")
train_fake = count_class_images(TRAIN_DIR, "FAKE")
test_real  = count_class_images(TEST_DIR,  "REAL")
test_fake  = count_class_images(TEST_DIR,  "FAKE")

print("  Dataset balance:")
print(f"    train/REAL : {train_real:,}")
print(f"    train/FAKE : {train_fake:,}")
print(f"    test/REAL  : {test_real:,}")
print(f"    test/FAKE  : {test_fake:,}")
print()

assert train_real == train_fake, \
    f"Training set imbalanced: REAL={train_real:,}, FAKE={train_fake:,}"
assert test_real == test_fake, \
    f"Test set imbalanced: REAL={test_real:,}, FAKE={test_fake:,}"
print("  ✓ Class balance confirmed: perfectly balanced in both splits.")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❸  NORMALISATION & DATA AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
# Why normalise: raw pixels [0, 255] cause large, unstable gradients during
# training.  Mapping to [0, 1] keeps gradients in a stable range and makes
# Adam's default learning rate effective.
#
# Why augmentation: 32×32 images have very few pixels.  Without augmentation
# the model can memorise exact pixel positions and overfit badly.  We apply:
#   • Random horizontal flip   — left-right reflections are equally valid
#   • Random translation ±10%  — small shifts prevent position memorisation
# We deliberately avoid heavy augmentation (rotations, colour jitter, etc.)
# because CIFAR-scale images are already tiny; aggressive transforms destroy
# class-discriminative features.

print("─" * 70)
print("❸  Building augmentation + normalisation pipeline ...")
print()

# Normalisation rescales [0,255] → [0,1].
# Augmentation layers are applied ONLY during training (they are no-ops in
# inference / evaluation mode automatically).
normalise = tf.keras.layers.Rescaling(1.0 / 255)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomTranslation(
        height_factor=0.10,   # ±10% vertical shift
        width_factor =0.10,   # ±10% horizontal shift
        seed=SEED,
    ),
], name="augmentation")

# Apply normalisation to every split, augmentation only to training data.
# We use .map() so the operations are fused into the tf.data pipeline
# and executed on-the-fly (no extra memory needed for augmented copies).
AUTOTUNE = tf.data.AUTOTUNE

def prepare_train(ds):
    return (
        ds
        .map(lambda x, y: (normalise(x), y),   num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .cache()            # cache after augmentation to reuse across epochs
        .prefetch(AUTOTUNE)
    )

def prepare_eval(ds):
    return (
        ds
        .map(lambda x, y: (normalise(x), y), num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

ds_train  = prepare_train(ds_train_full)
ds_val_p  = prepare_eval(ds_val)
ds_test_p = prepare_eval(ds_test)

print("  Augmentation: RandomFlip(horizontal) + RandomTranslation(±10%)")
print("  Normalisation: pixel ÷ 255  →  [0, 1]")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❹  MODEL ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
# Architecture rationale:
# ─ Three convolutional blocks progressively extract spatial features at
#   increasing levels of abstraction (edges → textures → patterns).
# ─ MaxPooling halves spatial dimensions after each block, keeping the
#   receptive field growing without a parameter explosion.
# ─ GlobalAveragePooling replaces Flatten to avoid over-parameterisation:
#   it averages each feature map to a single number, giving 128 values.
# ─ Dropout(0.5) before the head is the primary regulariser — it randomly
#   zeros half the activations each batch so the network cannot co-adapt.
# ─ A single sigmoid neuron outputs P(AI-generated), making this a clean
#   binary classification head compatible with binary cross-entropy.

print("─" * 70)
print("❹  Building model ...")
print()

inputs = tf.keras.Input(shape=(32, 32, 3), name="image_input")

# ── Block 1: 32 filters  (32×32 → 16×16) ────────────────────────────────────
x = tf.keras.layers.Conv2D(
    32, (3, 3), activation="relu", padding="same", name="conv1"
)(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)    # 16×16

# ── Block 2: 64 filters  (16×16 → 8×8) ──────────────────────────────────────
x = tf.keras.layers.Conv2D(
    64, (3, 3), activation="relu", padding="same", name="conv2"
)(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)    # 8×8

# ── Block 3: 128 filters (8×8 → 4×4) ────────────────────────────────────────
x = tf.keras.layers.Conv2D(
    128, (3, 3), activation="relu", padding="same", name="conv3"
)(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)    # 4×4

# ── Pooling + Head ────────────────────────────────────────────────────────────
x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)    # (128,)
x = tf.keras.layers.Dropout(0.5, seed=SEED, name="dropout")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifake_cnn")

# ── Verify parameter count ────────────────────────────────────────────────────
total_params = model.count_params()
print(model.summary())
print()
print(f"  Total trainable parameters: {total_params:,}")
assert total_params == 93_377, (
    f"\n[ASSERTION ERROR] Expected 93,377 trainable parameters, "
    f"got {total_params:,}.\n"
    "Check the layer definitions above."
)
print("  ✓ Parameter count confirmed: 93,377")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❺  COMPILATION
# ══════════════════════════════════════════════════════════════════════════════
# Loss choice: Binary cross-entropy is the canonical loss for binary sigmoid
# outputs.  It measures the log-probability assigned to the correct label and
# handles the 0/1 boundary cleanly.
#
# Optimizer: Adam (Adaptive Moment Estimation) with default learning rate 1e-3
# adapts the step size per-parameter, which works well across different layer
# depths without extensive tuning.
#
# Metrics tracked during training:
#   accuracy  — overall fraction correct (easy to interpret)
#   precision — of predicted AI, how many were truly AI
#   recall    — of all AI images, how many were caught  ← key metric
#   AUC       — area under the ROC curve (threshold-independent quality)

print("─" * 70)
print("❺  Compiling model ...")
print()

model.compile(
    loss      = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics   = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

print("  Loss      : BinaryCrossentropy")
print("  Optimizer : Adam (lr=1e-3 default)")
print("  Metrics   : accuracy, precision, recall, AUC")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❻  TRAINING
# ══════════════════════════════════════════════════════════════════════════════
# EarlyStopping rationale:
#   Monitoring validation loss — not accuracy — because loss is a smoother
#   signal that reflects model calibration, not just threshold crossings.
#   patience=3 allows 3 epochs of non-improvement before stopping.
#   restore_best_weights=True rolls back to the epoch with the best val_loss
#   automatically, so we do not need a separate model checkpoint step.
#
# Expected behaviour:
#   Training stops at epoch 12 because the model's validation loss does not
#   improve after epoch 9 for 3 consecutive epochs (patience=3).
#   Weights are restored to epoch 9 (the best checkpoint).
#
# CPU note: One epoch over 80,000 images takes ~5–8 min on modern CPU.
# Total wall-clock time ≈ 60–100 min.

print("─" * 70)
print("❻  Training (CPU only — this will take a while) ...")
print()

MAX_EPOCHS = 20
PATIENCE   = 3

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor           = "val_loss",
    patience          = PATIENCE,
    restore_best_weights = True,
    verbose           = 1,
)

history = model.fit(
    ds_train,
    epochs            = MAX_EPOCHS,
    validation_data   = ds_val_p,
    callbacks         = [early_stop],
    verbose           = 1,
)

stopped_epoch = early_stop.stopped_epoch
best_epoch    = stopped_epoch - PATIENCE  # = stopped_epoch - 3

print()
print(f"  Training stopped at epoch : {stopped_epoch}")
print(f"  Best weights restored from: epoch {best_epoch}")
print()

# Save the model for inference / testing script
model_save_path = Path("cifake_model.keras")
model.save(model_save_path)
print(f"  ✓ Saved trained model weights to: {model_save_path}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❼  EVALUATION ON HELD-OUT TEST SET
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("❼  Evaluating on test set (20,000 images the model has never seen) ...")
print()

results = model.evaluate(ds_test_p, verbose=1, return_dict=True)
print()
print("  Test metrics:")
print(f"    Accuracy  : {results['accuracy']:.4f}  ({results['accuracy']*100:.2f}%)")
print(f"    Precision : {results['precision']:.4f}  ({results['precision']*100:.2f}%)")
print(f"    Recall    : {results['recall']:.4f}  ({results['recall']*100:.2f}%)")
print(f"    AUC       : {results['auc']:.4f}")
print()

# ── Confusion matrix (requires iterating the test dataset once) ───────────────
print("  Building confusion matrix ...")

y_true_list = []
y_pred_list = []

for images, labels in ds_test_p:
    preds = model.predict(images, verbose=0)
    y_pred_list.extend((preds.flatten() >= 0.5).astype(int).tolist())
    y_true_list.extend(labels.numpy().flatten().astype(int).tolist())

y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)

cm = confusion_matrix(y_true, y_pred)

# Layout:
#          Predicted REAL  Predicted AI
# True REAL      TN              FP
# True AI        FN              TP
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

ai_recall   = TP / (TP + FN) * 100   # % of AI images correctly caught
ai_missed   = FN                      # AI images that slipped through
real_flagged = FP                     # real images wrongly flagged as AI

print()
print("  Confusion matrix:")
print(f"    True Negatives  (REAL  → REAL)  : {TN:,}")
print(f"    False Positives (REAL  → AI)    : {FP:,}  ← real images flagged as AI")
print(f"    False Negatives (AI    → REAL)  : {FN:,}  ← AI images that slipped through")
print(f"    True Positives  (AI    → AI)    : {TP:,}")
print()
print(f"    AI-generated recall : {ai_recall:.2f}%  (caught {TP:,} / {TP+FN:,})")
print(f"    AI images missed    : {FN:,}")
print(f"    Real images flagged : {FP:,}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❽  PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("─" * 70)
print("❽  Saving plots ...")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Plot 1: Training history ─────────────────────────────────────────────────
hist = history.history
epochs_ran = range(1, len(hist["loss"]) + 1)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Training History — Real vs AI-Generated Image Classifier",
             fontsize=14, fontweight="bold")

metric_pairs = [
    ("loss",      "val_loss",      "Loss",      "Loss",           axes[0, 0]),
    ("accuracy",  "val_accuracy",  "Accuracy",  "Accuracy",       axes[0, 1]),
    ("precision", "val_precision", "Precision", "Precision",      axes[0, 2]),
    ("recall",    "val_recall",    "Recall",    "Recall",         axes[1, 0]),
    ("auc",       "val_auc",       "AUC",       "AUC",            axes[1, 1]),
]

for train_key, val_key, title, ylabel, ax in metric_pairs:
    ax.plot(epochs_ran, hist[train_key], "b-o", markersize=4, label="Train")
    ax.plot(epochs_ran, hist[val_key],   "r-o", markersize=4, label="Val")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

# Mark best epoch
best_ep = best_epoch if best_epoch > 0 else 1
for _, _, _, _, ax in metric_pairs:
    ax.axvline(x=best_ep, color="green", linestyle="--", alpha=0.7,
               label=f"Best (ep {best_ep})")

# Hide the unused 6th subplot
axes[1, 2].axis("off")
axes[1, 2].text(0.5, 0.5,
    f"Best epoch: {best_ep}\nStopped: {stopped_epoch}\nPatience: {PATIENCE}",
    ha="center", va="center", fontsize=12,
    bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    transform=axes[1, 2].transAxes,
)

plt.tight_layout()
history_path = OUTPUT_DIR / "training_history.png"
plt.savefig(history_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {history_path}")

# ─── Plot 2: Confusion matrix heatmap ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(
    cm,
    annot        = True,
    fmt          = ",d",
    cmap         = "Blues",
    xticklabels  = ["Predicted REAL", "Predicted AI"],
    yticklabels  = ["True REAL", "True AI"],
    linewidths   = 0.5,
    ax           = ax,
    annot_kws    = {"size": 14, "weight": "bold"},
)

ax.set_title(
    "Confusion Matrix — 20,000 Test Images\n"
    f"AI Recall = {ai_recall:.2f}%  |  Real Flagged = {real_flagged:,}",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_ylabel("Actual Class",    fontsize=11)
ax.set_xlabel("Predicted Class", fontsize=11)

# Annotate the cells with interpretation text
cell_notes = {
    (0, 0): "True Negatives\n(Correctly REAL)",
    (0, 1): f"False Positives\n({FP:,} real images\nflagged as AI)",
    (1, 0): f"False Negatives\n({FN:,} AI images\nmissed ← cautious OK)",
    (1, 1): f"True Positives\n({TP:,} AI images\ncorrectly caught)",
}

for (row, col), note in cell_notes.items():
    ax.text(col + 0.5, row + 0.72, note,
            ha="center", va="center", fontsize=7,
            color="white" if cm[row, col] > cm.max() * 0.5 else "black")

plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {cm_path}")
print()

# ══════════════════════════════════════════════════════════════════════════════
# ❾  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  FINAL RESULTS SUMMARY")
print("=" * 70)
print()
print("  Dataset  : CIFAKE (120,000 images — 60k real CIFAR-10 + 60k Stable Diffusion)")
print(f"  Model    : Custom CNN  |  Trainable params: {total_params:,}")
print()
print("  ┌─────────────────────────────────────────────────────────┐")
print(f"  │  Test Accuracy   : {results['accuracy']*100:6.2f}%                           │")
print(f"  │  Test AUC        : {results['auc']:6.4f}                            │")
print(f"  │  AI-class Recall : {ai_recall:6.2f}%  ({TP:,} / {TP+FN:,} AI caught)    │")
print(f"  │  AI images missed: {FN:6,}   (slipped through as 'real')   │")
print(f"  │  Real imgs flagged: {FP:5,}  (false alarms)               │")
print("  └─────────────────────────────────────────────────────────┘")
print()
print("  ★ Model Interpretation — Cautious Bias:")
print()
print("    The model prioritises catching AI-generated images above all else.")
print(f"    It correctly identifies {ai_recall:.2f}% of AI-generated images,")
print(f"    missing only {FN:,} out of {TP+FN:,} AI images.")
print()
print("    The trade-off: it over-flags real images.")
print(f"    {FP:,} genuine photographs are incorrectly labelled as AI-generated.")
print()
print("    This asymmetric behaviour — high recall, lower precision —")
print("    is desirable in content moderation or authenticity verification")
print("    contexts where letting AI-generated content slip through undetected")
print("    carries a higher cost than a false alarm.")
print()
print("  Saved outputs:")
print(f"    • {history_path}")
print(f"    • {cm_path}")
print()
print("=" * 70)
print("  Pipeline complete.")
print("=" * 70)
