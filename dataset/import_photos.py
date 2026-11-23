"""Copy phone photos into dataset/raw as card_001.jpg, card_002.jpg, ... (capture order).

Usage (from the repo root):
    python dataset/import_photos.py "C:\\path\\to\\folder\\with\\phone\\photos"
The originals are copied, never moved or changed. The old-name -> new-name table is
written to dataset/photo_name_map.csv.
"""
import csv
import glob
import os
import shutil
import sys

DEST = "dataset/raw"


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            'Usage: python dataset/import_photos.py "<folder with photos>"')
    source = sys.argv[1]
    paths = sorted(
        p for p in glob.glob(os.path.join(source, "*"))
        if p.lower().endswith((".jpg", ".jpeg"))
    )
    if not paths:
        raise SystemExit(f"No .jpg photos found in {source}")

    os.makedirs(DEST, exist_ok=True)
    if glob.glob(os.path.join(DEST, "card_*.jpg")):
        raise SystemExit(
            f"{DEST} already holds card_*.jpg files. Empty it first so nothing is overwritten.")

    with open("dataset/photo_name_map.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["new_name", "original_name"])
        for i, path in enumerate(paths, start=1):
            new_name = f"card_{i:03d}.jpg"
            shutil.copy2(path, os.path.join(DEST, new_name))
            writer.writerow([new_name, os.path.basename(path)])
    print(
        f"Copied {len(paths)} photos to {DEST} (card_001.jpg ... card_{len(paths):03d}.jpg)")
    print("Name map saved to dataset/photo_name_map.csv")


if __name__ == "__main__":
    main()
