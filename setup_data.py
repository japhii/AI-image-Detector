"""
setup_data.py
=============
Unzips the manually-downloaded Kaggle CIFAKE dataset and organises it into
the folder structure that train.py expects:

    data/
    ├── train/
    │   ├── REAL/   (50,000 images — used for training + validation)
    │   └── FAKE/   (50,000 images — used for training + validation)
    └── test/
        ├── REAL/   (10,000 images — held-out test set)
        └── FAKE/   (10,000 images — held-out test set)

Usage
-----
1. Download the dataset from Kaggle:
   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
2. Place the downloaded file in this folder and rename it to 'cifake.zip'
3. Run:  python setup_data.py
"""

import shutil
import zipfile
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────
ZIP_PATH = Path("cifake.zip")   # Where you placed the Kaggle download
DATA_DIR = Path("data")         # Destination folder

EXPECTED_TRAIN_REAL = 50_000
EXPECTED_TRAIN_FAKE = 50_000
EXPECTED_TEST_REAL  = 10_000
EXPECTED_TEST_FAKE  = 10_000


def find_dir(root: Path, *parts: str) -> Path:
    """
    Recursively search for a subfolder matching the given path parts.
    Tries an exact match first, then a case-insensitive match.
    This handles zip files that have a top-level wrapper folder.
    """
    target = str(Path(*parts))

    # Exact match
    candidates = list(root.rglob(target))

    if not candidates:
        # Case-insensitive fallback
        lower_parts = [p.lower() for p in parts]
        for path in root.rglob("*"):
            tail = path.parts[-len(parts):]
            if len(tail) == len(parts) and all(
                a.lower() == b for a, b in zip(tail, lower_parts)
            ):
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find '{target}' inside '{root}'. "
            "The zip may have an unexpected folder layout — "
            "check the Kaggle page for details."
        )

    return candidates[0]


def copy_folder(src: Path, dst: Path) -> None:
    """Copy all files from src into dst (creates dst if it does not exist)."""
    dst.mkdir(parents=True, exist_ok=True)
    for img in src.iterdir():
        if img.is_file():
            shutil.copy2(img, dst / img.name)


def count_images(folder: Path) -> int:
    """Return the number of image files in a folder (by extension)."""
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in image_extensions)


def main() -> None:
    # ── Step 1: Check that the zip file exists ────────────────────────────────
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] '{ZIP_PATH}' was not found in the current directory.\n\n"
            "Please download the CIFAKE dataset from:\n"
            "  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images\n\n"
            f"Then save it here as '{ZIP_PATH}'.\n"
        )

    # ── Step 2: Extract the zip into a temporary folder ───────────────────────
    extract_dir = Path("_cifake_raw")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    print(f"[1/3] Extracting '{ZIP_PATH}' → '{extract_dir}/' ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(extract_dir)
    print("      Done.\n")

    # ── Step 3: Locate the image folders inside the extracted archive ─────────
    src_train_real = find_dir(extract_dir, "train", "REAL")
    src_train_fake = find_dir(extract_dir, "train", "FAKE")
    src_test_real  = find_dir(extract_dir, "test",  "REAL")
    src_test_fake  = find_dir(extract_dir, "test",  "FAKE")

    # ── Step 4: Copy images into the expected data/ layout ───────────────────
    print("[2/3] Copying images into data/ ...")
    copy_folder(src_train_real, DATA_DIR / "train" / "REAL")
    copy_folder(src_train_fake, DATA_DIR / "train" / "FAKE")
    copy_folder(src_test_real,  DATA_DIR / "test"  / "REAL")
    copy_folder(src_test_fake,  DATA_DIR / "test"  / "FAKE")
    print("      Done.\n")

    # ── Step 5: Count images and warn if numbers don't match ──────────────────
    print("[3/3] Verifying image counts ...")

    counts = {
        "train/REAL": count_images(DATA_DIR / "train" / "REAL"),
        "train/FAKE": count_images(DATA_DIR / "train" / "FAKE"),
        "test/REAL":  count_images(DATA_DIR / "test"  / "REAL"),
        "test/FAKE":  count_images(DATA_DIR / "test"  / "FAKE"),
    }

    expected = {
        "train/REAL": EXPECTED_TRAIN_REAL,
        "train/FAKE": EXPECTED_TRAIN_FAKE,
        "test/REAL":  EXPECTED_TEST_REAL,
        "test/FAKE":  EXPECTED_TEST_FAKE,
    }

    for path, count in counts.items():
        status = "✓" if count == expected[path] else "⚠️ "
        print(f"      {status}  data/{path}: {count:,} images (expected {expected[path]:,})")

    # ── Step 6: Clean up the temporary extraction folder ─────────────────────
    shutil.rmtree(extract_dir)

    print("\n✓ Dataset ready. You can now run:  python train.py\n")


if __name__ == "__main__":
    main()
