"""
setup_data.py
=============
Unzips the manually-downloaded Kaggle CIFAKE dataset and organises it into the
expected folder structure:

    data/
    ├── train/
    │   ├── REAL/   (50,000 images — train + validation source)
    │   └── FAKE/   (50,000 images — train + validation source)
    └── test/
        ├── REAL/   (10,000 images)
        └── FAKE/   (10,000 images)

Usage
-----
1. Download the zip from Kaggle:
   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
2. Rename / place it in this project folder as  'cifake.zip'
3. Run:  python setup_data.py
"""

import os
import shutil
import zipfile
import glob
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
ZIP_PATH   = Path("cifake.zip")          # where you placed the download
DATA_DIR   = Path("data")               # output root

EXPECTED_TRAIN_REAL = 50_000
EXPECTED_TRAIN_FAKE = 50_000
EXPECTED_TEST_REAL  = 10_000
EXPECTED_TEST_FAKE  = 10_000


def main() -> None:
    # ── 1. Verify zip exists ──────────────────────────────────────────────────
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"\n[ERROR] '{ZIP_PATH}' not found in the current directory.\n"
            "Please download the CIFAKE dataset from:\n"
            "  https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images\n"
            f"and save it as '{ZIP_PATH}' in:\n"
            f"  {Path.cwd()}\n"
        )

    # ── 2. Extract zip ────────────────────────────────────────────────────────
    extract_dir = Path("_cifake_raw")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    print(f"[1/3] Extracting '{ZIP_PATH}' → '{extract_dir}/' ...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(extract_dir)
    print("      Done.")

    # ── 3. Locate the actual image folders inside the extract ─────────────────
    # The Kaggle zip typically has structure: train/REAL, train/FAKE, test/REAL, test/FAKE
    # We search for them flexibly in case there is a top-level wrapper folder.
    def find_dir(root: Path, *parts: str) -> Path:
        """Recursively search for a sub-path under root."""
        candidates = list(root.rglob(str(Path(*parts))))
        if not candidates:
            # Try case-insensitive search
            lower_parts = [p.lower() for p in parts]
            for p in root.rglob("*"):
                if p.parts[-len(parts):] and all(
                    a.lower() == b for a, b in zip(p.parts[-len(parts):], lower_parts)
                ):
                    candidates.append(p)
        if not candidates:
            raise FileNotFoundError(
                f"Cannot find '{Path(*parts)}' under '{root}'. "
                "Unexpected zip structure — check the Kaggle page for folder layout."
            )
        return candidates[0]

    src_train_real = find_dir(extract_dir, "train", "REAL")
    src_train_fake = find_dir(extract_dir, "train", "FAKE")
    src_test_real  = find_dir(extract_dir, "test",  "REAL")
    src_test_fake  = find_dir(extract_dir, "test",  "FAKE")

    # ── 4. Copy into data/ with the correct layout ────────────────────────────
    print("[2/3] Copying images into data/ ...")

    def copy_folder(src: Path, dst: Path) -> None:
        dst.mkdir(parents=True, exist_ok=True)
        for img in src.iterdir():
            if img.is_file():
                shutil.copy2(img, dst / img.name)

    copy_folder(src_train_real, DATA_DIR / "train" / "REAL")
    copy_folder(src_train_fake, DATA_DIR / "train" / "FAKE")
    copy_folder(src_test_real,  DATA_DIR / "test"  / "REAL")
    copy_folder(src_test_fake,  DATA_DIR / "test"  / "FAKE")
    print("      Done.")

    # ── 5. Verify counts ──────────────────────────────────────────────────────
    print("[3/3] Verifying image counts ...")

    def count_images(folder: Path) -> int:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
        return sum(1 for f in folder.iterdir() if f.suffix.lower() in exts)

    counts = {
        "train/REAL": count_images(DATA_DIR / "train" / "REAL"),
        "train/FAKE": count_images(DATA_DIR / "train" / "FAKE"),
        "test/REAL":  count_images(DATA_DIR / "test"  / "REAL"),
        "test/FAKE":  count_images(DATA_DIR / "test"  / "FAKE"),
    }

    for path, count in counts.items():
        print(f"      data/{path}: {count:,} images")

    # Soft warnings (the Kaggle zip ships 50k / 50k / 10k / 10k)
    if counts["train/REAL"] != EXPECTED_TRAIN_REAL:
        print(f"  [WARN] Expected {EXPECTED_TRAIN_REAL:,} train/REAL images, got {counts['train/REAL']:,}")
    if counts["train/FAKE"] != EXPECTED_TRAIN_FAKE:
        print(f"  [WARN] Expected {EXPECTED_TRAIN_FAKE:,} train/FAKE images, got {counts['train/FAKE']:,}")
    if counts["test/REAL"] != EXPECTED_TEST_REAL:
        print(f"  [WARN] Expected {EXPECTED_TEST_REAL:,} test/REAL  images, got {counts['test/REAL']:,}")
    if counts["test/FAKE"] != EXPECTED_TEST_FAKE:
        print(f"  [WARN] Expected {EXPECTED_TEST_FAKE:,} test/FAKE  images, got {counts['test/FAKE']:,}")

    # ── 6. Clean up temp extraction ───────────────────────────────────────────
    shutil.rmtree(extract_dir)
    print("\n✓ Dataset ready. Run:  python train.py")


if __name__ == "__main__":
    main()
