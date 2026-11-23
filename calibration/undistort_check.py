"""Undistort one photo with the saved calibration and save a side-by-side check image.

Usage (from the repo root):
    python calibration/undistort_check.py path/to/photo.jpg
Writes calibration/undistort_check.jpg (original left, undistorted right, with a
straight reference grid drawn on both so you can see whether edges stay straight).
"""
import argparse

import cv2
import numpy as np

READ_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--params", default="calibration/camera_params.npz")
    parser.add_argument("--out", default="calibration/undistort_check.jpg")
    args = parser.parse_args()

    params = np.load(args.params)
    camera_matrix, dist = params["camera_matrix"], params["dist_coeffs"]
    expected_size = tuple(int(v) for v in params["image_size"])

    image = cv2.imread(args.image, READ_FLAGS)
    if image is None:
        raise SystemExit(f"Could not read {args.image}")
    size = (image.shape[1], image.shape[0])
    if size != expected_size:
        raise SystemExit(f"Image is {size[0]}x{size[1]} but the calibration is for "
                         f"{expected_size[0]}x{expected_size[1]}. Use the same camera mode.")

    undistorted = cv2.undistort(image, camera_matrix, dist)

    def with_grid(img):
        img = cv2.resize(img, (1000, int(1000 * img.shape[0] / img.shape[1])))
        h, w = img.shape[:2]
        for i in range(1, 8):
            cv2.line(img, (0, i * h // 8), (w, i * h // 8), (0, 255, 0), 1)
            cv2.line(img, (i * w // 8, 0), (i * w // 8, h), (0, 255, 0), 1)
        return img

    side_by_side = np.hstack([with_grid(image), with_grid(undistorted)])
    cv2.imwrite(args.out, side_by_side)
    print(f"Saved {args.out}  (left: original, right: undistorted)")


if __name__ == "__main__":
    main()
