"""Calibrate the phone camera from ChArUco board photos.

Usage (from the repo root):
    python calibration/calibrate.py
Reads  calibration/images/*.jpg
Writes calibration/camera_params.npz and calibration/camera_params.json

Images are read in the sensor's own orientation (EXIF rotation ignored), so a
photo taken with the phone held sideways still belongs to the same camera.
The most common image size is used; photos of any other size are skipped.
"""
import argparse
import glob
import json
import os
from collections import Counter

import cv2
import numpy as np

SQUARES_X, SQUARES_Y = 5, 7        # board as printed: 5 x 7 squares
# replace with your ruler-checked values if they differ
SQUARE_MM, MARKER_MM = 25.0, 18.0
MIN_CORNERS = 12                   # skip photos where fewer ChArUco corners are found
READ_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def run_calibration(obj_points, img_points, image_size, flags):
    """Calibrate and return (rms, camera matrix, distortion, per-image errors in px)."""
    rms, camera_matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, flags=flags
    )
    per_image = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist)
        diff = img_points[i].reshape(-1, 2) - projected.reshape(-1, 2)
        per_image.append(float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))))
    return rms, camera_matrix, dist, per_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="calibration/images")
    parser.add_argument("--out", default="calibration/camera_params")
    parser.add_argument("--min-images", type=int, default=20)
    parser.add_argument("--max-error", type=float, default=None,
                        help="drop photos whose reprojection error is above this (px), then recalibrate")
    parser.add_argument("--simple", action="store_true",
                        help="fit only k1, k2 (fix k3, no tangential terms): steadier for a phone lens")
    args = parser.parse_args()
    flags = (cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST) if args.simple else 0

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_MM /
        1000.0, MARKER_MM / 1000.0, aruco_dict
    )
    detector = cv2.aruco.CharucoDetector(board)

    paths = sorted(
        p for p in glob.glob(os.path.join(args.images, "*"))
        if p.lower().endswith((".jpg", ".jpeg"))
    )
    print(f"Found {len(paths)} images in {os.path.abspath(args.images)}")

    detections = []  # (path, size, object points, image points)
    skipped = []
    for path in paths:
        name = os.path.basename(path)
        image = cv2.imread(path, READ_FLAGS)
        if image is None:
            skipped.append((path, "could not be read"))
            print(f"  [skip] {name}: could not be read")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]  # (width, height)
        corners, ids, _, _ = detector.detectBoard(gray)
        n = 0 if ids is None else len(ids)
        if n < MIN_CORNERS:
            skipped.append((path, f"only {n} board corners found"))
            print(f"  [skip] {name}: only {n} board corners found")
            continue
        obj, img = board.matchImagePoints(corners, ids)
        detections.append((path, size, obj, img))
        print(f"  [ok]   {name}: {n} board corners, size {size[0]}x{size[1]}")

    if not detections:
        raise SystemExit("No usable images, cannot calibrate.")

    image_size = Counter(d[1] for d in detections).most_common(1)[0][0]
    used, obj_points, img_points = [], [], []
    for path, size, obj, img in detections:
        if size != image_size:
            skipped.append((path, f"size {size} differs from {image_size}"))
            print(
                f"  [skip] {os.path.basename(path)}: size {size} differs from {image_size}")
            continue
        used.append(path)
        obj_points.append(obj)
        img_points.append(img)

    if len(used) < 3:
        raise SystemExit(
            f"Only {len(used)} usable images ({len(skipped)} skipped), cannot calibrate.")

    rms, camera_matrix, dist, per_image = run_calibration(
        obj_points, img_points, image_size, flags)

    dropped = []
    if args.max_error is not None:
        keep = [i for i, e in enumerate(per_image) if e <= args.max_error]
        candidates = [(used[i], per_image[i])
                      for i in range(len(used)) if i not in keep]
        if candidates and len(keep) >= 3:
            dropped = candidates
            print(
                f"\nDropping {len(dropped)} photos with error above {args.max_error} px:")
            for path, e in dropped:
                print(f"  {os.path.basename(path)}: {e:.3f}")
            used = [used[i] for i in keep]
            obj_points = [obj_points[i] for i in keep]
            img_points = [img_points[i] for i in keep]
            rms, camera_matrix, dist, per_image = run_calibration(
                obj_points, img_points, image_size, flags)

    print(f"\nImage size: {image_size[0]} x {image_size[1]}")
    print(
        f"Images used: {len(used)}   skipped: {len(skipped)}   dropped for high error: {len(dropped)}")
    if len(used) < args.min_images:
        print(
            f"WARNING: fewer than {args.min_images} images, the task needs at least 20.")

    print("\nPer-image reprojection error (px):")
    for path, err in zip(used, per_image):
        print(f"  {os.path.basename(path)}: {err:.3f}")
    print(f"\nMean / median / worst per-image error: "
          f"{np.mean(per_image):.3f} / {np.median(per_image):.3f} / {np.max(per_image):.3f} px")

    print(f"\nOverall RMS reprojection error: {rms:.4f} px")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients (k1 k2 p1 p2 k3): {dist.ravel()}")

    np.savez(args.out + ".npz", camera_matrix=camera_matrix, dist_coeffs=dist,
             image_size=np.array(image_size))
    with open(args.out + ".json", "w") as f:
        json.dump({
            "image_size": list(image_size),
            "images_used": len(used),
            "images_dropped_for_high_error": [os.path.basename(p) for p, _ in dropped],
            "simple_distortion_model": bool(args.simple),
            "rms_reprojection_error_px": float(rms),
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist.ravel().tolist(),
            "opencv_version": cv2.__version__,
        }, f, indent=2)
    print(f"\nSaved {args.out}.npz and {args.out}.json")


if __name__ == "__main__":
    main()
