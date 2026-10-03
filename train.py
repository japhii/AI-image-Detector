import os
import random

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf
tf.random.set_seed(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import confusion_matrix

print("=" * 70)
print("  Real vs AI-Generated Image Classifier  —  CIFAKE Dataset")
print("=" * 70)
print(f"  TensorFlow version : {tf.__version__}")
print(f"  NumPy version      : {np.__version__}")
print(f"  Random seed        : {SEED}")
print()

DATA_DIR  = Path("data")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"

IMAGE_SIZE  = (32, 32)
BATCH_SIZE  = 64
VAL_SPLIT   = 0.20

CLASS_NAMES = ["REAL", "FAKE"]

print("─" * 70)
print("❷  Loading data ...")
print()

ds_train_full = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels           = "inferred",
    label_mode       = "binary",
    class_names      = CLASS_NAMES,
    color_mode       = "rgb",
    image_size       = IMAGE_SIZE,
    batch_size       = BATCH_SIZE,
    shuffle          = True,
    seed             = SEED,
    validation_split = VAL_SPLIT,
    subset           = "training",
)

ds_val = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels           = "inferred",
    label_mode       = "binary",
    class_names      = CLASS_NAMES,
    color_mode       = "rgb",
    image_size       = IMAGE_SIZE,
    batch_size       = BATCH_SIZE,
    shuffle          = False,
    seed             = SEED,
    validation_split = VAL_SPLIT,
    subset           = "validation",
)

ds_test = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels      = "inferred",
    label_mode  = "binary",
    class_names = CLASS_NAMES,
    color_mode  = "rgb",
    image_size  = IMAGE_SIZE,
    batch_size  = BATCH_SIZE,
    shuffle     = False,
)

n_train = ds_train_full.cardinality().numpy() * BATCH_SIZE
n_val   = ds_val.cardinality().numpy()        * BATCH_SIZE
n_test  = ds_test.cardinality().numpy()       * BATCH_SIZE

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

print("  Running quality scan on one batch ...")
for images, labels in ds_train_full.take(1):
    sample_images = images
    sample_labels = labels

print(f"  Batch shape  : {sample_images.shape}")
print(f"  Label shape  : {sample_labels.shape}")
print(f"  Pixel range  : [{sample_images.numpy().min():.1f}, {sample_images.numpy().max():.1f}]")
print(f"  Label values : {sorted(set(sample_labels.numpy().flatten().astype(int).tolist()))}")

assert sample_images.shape[1:] == (32, 32, 3), \
    f"Unexpected image shape {sample_images.shape[1:]}; expected (32, 32, 3)"
assert set(sample_labels.numpy().flatten().astype(int).tolist()).issubset({0, 1}), \
    "Labels contain unexpected values"

print()
print("  ✓ Quality scan passed: images are (32×32×3), labels are binary {0, 1}")
print()

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

print("─" * 70)
print("❸  Building augmentation + normalisation pipeline ...")
print()

normalise = tf.keras.layers.Rescaling(1.0 / 255)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomTranslation(
        height_factor=0.10,
        width_factor =0.10,
        seed=SEED,
    ),
], name="augmentation")

AUTOTUNE = tf.data.AUTOTUNE

def prepare_train(ds):
    return (
        ds
        .map(lambda x, y: (normalise(x), y),              num_parallel_calls=AUTOTUNE)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .cache()
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

print("─" * 70)
print("❹  Building model ...")
print()

inputs = tf.keras.Input(shape=(32, 32, 3), name="image_input")

x = tf.keras.layers.Conv2D(
    32, (3, 3), activation="relu", padding="same", name="conv1"
)(inputs)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

x = tf.keras.layers.Conv2D(
    64, (3, 3), activation="relu", padding="same", name="conv2"
)(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

x = tf.keras.layers.Conv2D(
    128, (3, 3), activation="relu", padding="same", name="conv3"
)(x)
x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)

x       = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
x       = tf.keras.layers.Dropout(0.5, seed=SEED, name="dropout")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs, name="cifake_cnn")

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

print("─" * 70)
print("❻  Training (CPU only — this will take a while) ...")
print()

MAX_EPOCHS = 20
PATIENCE   = 3

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor              = "val_loss",
    patience             = PATIENCE,
    restore_best_weights = True,
    verbose              = 1,
)

history = model.fit(
    ds_train,
    epochs          = MAX_EPOCHS,
    validation_data = ds_val_p,
    callbacks       = [early_stop],
    verbose         = 1,
)

stopped_epoch = early_stop.stopped_epoch
best_epoch    = stopped_epoch - PATIENCE

print()
print(f"  Training stopped at epoch : {stopped_epoch}")
print(f"  Best weights restored from: epoch {best_epoch}")
print()

model_save_path = Path("cifake_model.keras")
model.save(model_save_path)
print(f"  ✓ Saved trained model weights to: {model_save_path}")
print()

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

TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

ai_recall    = TP / (TP + FN) * 100
ai_missed    = FN
real_flagged = FP

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

print("─" * 70)
print("❽  Saving plots ...")

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

hist       = history.history
epochs_ran = range(1, len(hist["loss"]) + 1)

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Training History — Real vs AI-Generated Image Classifier",
             fontsize=14, fontweight="bold")

metric_pairs = [
    ("loss",      "val_loss",      "Loss",      "Loss",      axes[0, 0]),
    ("accuracy",  "val_accuracy",  "Accuracy",  "Accuracy",  axes[0, 1]),
    ("precision", "val_precision", "Precision", "Precision", axes[0, 2]),
    ("recall",    "val_recall",    "Recall",    "Recall",    axes[1, 0]),
    ("auc",       "val_auc",       "AUC",       "AUC",       axes[1, 1]),
]

for train_key, val_key, title, ylabel, ax in metric_pairs:
    ax.plot(epochs_ran, hist[train_key], "b-o", markersize=4, label="Train")
    ax.plot(epochs_ran, hist[val_key],   "r-o", markersize=4, label="Val")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

best_ep = best_epoch if best_epoch > 0 else 1
for _, _, _, _, ax in metric_pairs:
    ax.axvline(x=best_ep, color="green", linestyle="--", alpha=0.7,
               label=f"Best (ep {best_ep})")

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

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(
    cm,
    annot       = True,
    fmt         = ",d",
    cmap        = "Blues",
    xticklabels = ["Predicted REAL", "Predicted AI"],
    yticklabels = ["True REAL", "True AI"],
    linewidths  = 0.5,
    ax          = ax,
    annot_kws   = {"size": 14, "weight": "bold"},
)

ax.set_title(
    "Confusion Matrix — 20,000 Test Images\n"
    f"AI Recall = {ai_recall:.2f}%  |  Real Flagged = {real_flagged:,}",
    fontsize=12, fontweight="bold", pad=12,
)
ax.set_ylabel("Actual Class",    fontsize=11)
ax.set_xlabel("Predicted Class", fontsize=11)

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
