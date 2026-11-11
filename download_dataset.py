#!/usr/bin/env python3
"""
download_dataset.py — Download and prepare the LC25000 Lung Cancer
Histopathological Image Dataset.

Dataset:
    Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM.
    Lung and Colon Cancer Histopathological Image Dataset (LC25000).
    arXiv:1912.12142v1 [eess.IV], 2019.

The full LC25000 has 5 classes (25,000 images):
    - lung_n      (5,000) — benign/normal lung tissue
    - lung_aca    (5,000) — lung adenocarcinoma
    - lung_scc    (5,000) — lung squamous cell carcinoma
    - colon_n     (5,000) — benign/normal colon tissue
    - colon_aca   (5,000) — lung adenocarcinoma

This script downloads from Kaggle and organises the lung-cancer-only
subset (15,000 images, 3 classes) into the folder layout expected by
our data pipeline.

Usage:
    # Method 1 — Kaggle API (recommended, needs kaggle.json credentials)
    python download_dataset.py

    # Method 2 — Manual download
    python download_dataset.py --from-zip /path/to/archive.zip

    # Include colon classes too (all 5 classes, 25k images)
    python download_dataset.py --include-colon

Prerequisites (for automatic download):
    pip install kaggle
    # Place your kaggle.json in ~/.kaggle/kaggle.json
    # Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Kaggle dataset identifier
KAGGLE_DATASET = "andrewmvd/lung-and-colon-cancer-histopathological-images"

# Classes we keep for lung cancer detection
LUNG_CLASSES = {"lung_n", "lung_aca", "lung_scc"}
ALL_CLASSES = {"lung_n", "lung_aca", "lung_scc", "colon_n", "colon_aca"}

# Human-readable names for reporting
CLASS_LABELS = {
    "lung_n": "Normal Lung Tissue",
    "lung_aca": "Lung Adenocarcinoma",
    "lung_scc": "Lung Squamous Cell Carcinoma",
    "colon_n": "Normal Colon Tissue",
    "colon_aca": "Colon Adenocarcinoma",
}


def download_from_kaggle(dest_dir: Path) -> Path:
    """Download the dataset zip from Kaggle using the Kaggle API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print(
            "\n❌ The 'kaggle' Python package is not installed.\n"
            "   Install it with:  pip install kaggle\n"
            "   Then place your API token at ~/.kaggle/kaggle.json\n"
            "   Or set KAGGLE_USERNAME and KAGGLE_KEY env vars.\n"
            "\n   Alternatively, download the dataset manually from:\n"
            "   https://www.kaggle.com/datasets/andrewmvd/"
            "lung-and-colon-cancer-histopathological-images\n"
            "   and re-run with:  python download_dataset.py --from-zip <path>\n"
        )
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()

    print(f"📥 Downloading {KAGGLE_DATASET} from Kaggle...")
    print(f"   Destination: {dest_dir}")

    api.dataset_download_files(KAGGLE_DATASET, path=str(dest_dir), unzip=False)

    # The downloaded file name
    zip_name = "lung-and-colon-cancer-histopathological-images.zip"
    zip_path = dest_dir / zip_name

    if not zip_path.exists():
        # Kaggle sometimes uses different names; find the zip
        zips = list(dest_dir.glob("*.zip"))
        if zips:
            zip_path = zips[0]
        else:
            print("❌ Download completed but no zip file found.")
            sys.exit(1)

    print(f"✅ Downloaded: {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")
    return zip_path


def extract_and_organise(
    zip_path: Path,
    output_dir: Path,
    include_colon: bool = False,
) -> None:
    """Extract the zip and organise into the expected folder structure."""
    keep_classes = ALL_CLASSES if include_colon else LUNG_CLASSES

    print(f"\n📦 Extracting {zip_path.name}...")
    temp_dir = output_dir / "_temp_extract"
    temp_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(temp_dir)

    print("📁 Organising into class folders...")

    # The zip usually contains: lung_colon_image_set/lung_image_sets/... and colon_image_sets/...
    # We need to find the actual image directories
    image_dirs: dict[str, Path] = {}

    for dirpath, dirnames, filenames in os.walk(temp_dir):
        dir_name = os.path.basename(dirpath).lower()
        # Map common folder names to our class names
        name_map = {
            "lung_n": "lung_n",
            "lung_benign_tissue": "lung_n",
            "lung_aca": "lung_aca",
            "lung_adenocarcinoma": "lung_aca",
            "lung_scc": "lung_scc",
            "lung_squamous_cell_carcinoma": "lung_scc",
            "colon_n": "colon_n",
            "colon_benign_tissue": "colon_n",
            "colon_aca": "colon_aca",
            "colon_adenocarcinoma": "colon_aca",
        }

        if dir_name in name_map:
            mapped = name_map[dir_name]
            if mapped in keep_classes:
                image_dirs[mapped] = Path(dirpath)

    if not image_dirs:
        # Fallback: search for directories containing many images
        print("   ⚠️  Standard folder names not found, scanning for image directories...")
        for dirpath, _, filenames in os.walk(temp_dir):
            imgs = [f for f in filenames if f.lower().endswith((".jpeg", ".jpg", ".png"))]
            if len(imgs) > 100:
                dir_name = os.path.basename(dirpath).lower()
                print(f"   Found: {dirpath} ({len(imgs)} images)")

    # Copy images to final structure
    total_copied = 0
    for class_name, src_dir in sorted(image_dirs.items()):
        dest_class_dir = output_dir / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)

        imgs = [
            f for f in src_dir.iterdir()
            if f.suffix.lower() in {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
        ]

        for img in imgs:
            shutil.copy2(img, dest_class_dir / img.name)

        total_copied += len(imgs)
        print(f"   ✅ {class_name:20s} → {len(imgs):,} images  ({CLASS_LABELS.get(class_name, '')})")

    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"\n✅ Dataset ready: {total_copied:,} images in {len(image_dirs)} classes")
    print(f"   Location: {output_dir}")


def verify_dataset(data_dir: Path) -> bool:
    """Verify the dataset structure and print a summary."""
    print(f"\n🔍 Verifying dataset at: {data_dir}\n")

    if not data_dir.exists():
        print(f"   ❌ Directory not found: {data_dir}")
        return False

    valid_ext = {".jpeg", ".jpg", ".png", ".tif", ".tiff"}
    total = 0
    classes_found = []

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith((".", "_")):
            continue

        imgs = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_ext]
        count = len(imgs)
        total += count
        classes_found.append((class_dir.name, count))

        label = CLASS_LABELS.get(class_dir.name, "")
        status = "✅" if count > 0 else "❌"
        print(f"   {status} {class_dir.name:25s}  {count:>6,} images  {label}")

    print(f"\n   {'Total':25s}  {total:>6,} images")
    print(f"   {'Classes':25s}  {len(classes_found):>6}")

    if total == 0:
        print("\n   ❌ No images found! Check the directory structure.")
        return False

    print("\n   ✅ Dataset verification passed!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare the LC25000 Lung Cancer Histopathology Dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_dataset.py                         # Download via Kaggle API
  python download_dataset.py --from-zip archive.zip  # Extract from a local zip
  python download_dataset.py --include-colon         # Include colon classes too
  python download_dataset.py --verify-only           # Just check existing data
        """,
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data",
        help="Output directory for the organised dataset (default: ./data)",
    )
    parser.add_argument(
        "--from-zip", type=str, default=None,
        help="Path to a manually downloaded zip file (skips Kaggle download)",
    )
    parser.add_argument(
        "--include-colon", action="store_true",
        help="Include colon cancer classes (5 classes instead of 3)",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Only verify the existing dataset structure, don't download",
    )
    parser.add_argument(
        "--keep-zip", action="store_true",
        help="Keep the downloaded zip file after extraction",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir).resolve()

    if args.verify_only:
        verify_dataset(output_dir)
        return

    print("=" * 60)
    print("  LC25000 — Lung Cancer Histopathology Dataset Setup")
    print("=" * 60)

    n_classes = 5 if args.include_colon else 3
    n_images = 25000 if args.include_colon else 15000
    print(f"\n  Classes: {n_classes} ({'lung + colon' if args.include_colon else 'lung only'})")
    print(f"  Images:  ~{n_images:,}")
    print(f"  Output:  {output_dir}\n")

    # Step 1: Get the zip file
    if args.from_zip:
        zip_path = Path(args.from_zip).resolve()
        if not zip_path.exists():
            print(f"❌ Zip file not found: {zip_path}")
            sys.exit(1)
        print(f"📦 Using provided zip: {zip_path}")
    else:
        zip_path = download_from_kaggle(output_dir.parent)

    # Step 2: Extract and organise
    extract_and_organise(zip_path, output_dir, include_colon=args.include_colon)

    # Step 3: Clean up zip if requested
    if not args.keep_zip and not args.from_zip:
        print(f"\n🗑️  Removing zip file: {zip_path}")
        zip_path.unlink(missing_ok=True)

    # Step 4: Verify
    verify_dataset(output_dir)

    print("\n" + "=" * 60)
    print("  Next step: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
