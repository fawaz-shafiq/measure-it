"""Rotate portrait photos if needed and undistort all raw dataset images.

Usage (from the repo root):
    python dataset/undistort_dataset.py
Reads  dataset/raw/*.jpg and calibration/camera_params.npz
Writes dataset/undistorted/*.jpg with the same filenames.
"""
import glob
import os

import cv2
import numpy as np

RAW_DIR = "dataset/raw"
OUT_DIR = "dataset/undistorted"
PARAMS_PATH = "calibration/camera_params.npz"
READ_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def main():
    params = np.load(PARAMS_PATH)
    camera_matrix = params["camera_matrix"]
    dist_coeffs = params["dist_coeffs"]
    calib_size = tuple(int(v) for v in params["image_size"])  # (width, height)

    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(
        p for p in glob.glob(os.path.join(RAW_DIR, "*"))
        if p.lower().endswith((".jpg", ".jpeg"))
    )
    if not paths:
        raise SystemExit(f"No images found in {RAW_DIR}")

    rotated = 0
    saved = 0
    for path in paths:
        name = os.path.basename(path)
        img = cv2.imread(path, READ_FLAGS)
        if img is None:
            print(f"  [skip] {name}: could not be read")
            continue

        h, w = img.shape[:2]
        if (w, h) == (calib_size[1], calib_size[0]):
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            rotated += 1
            h, w = img.shape[:2]

        if (w, h) != calib_size:
            print(f"  [skip] {name}: size {w}x{h} != calibration {calib_size[0]}x{calib_size[1]}")
            continue

        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)
        cv2.imwrite(os.path.join(OUT_DIR, name), undistorted)
        saved += 1

    print(f"Saved {saved} undistorted images to {OUT_DIR} ({rotated} rotated 90 deg CW).")
    print(f"Output size: {calib_size[0]}x{calib_size[1]}")


if __name__ == "__main__":
    main()
