"""Check every photo in dataset/raw before labelling.

Usage (from the repo root):
    python dataset/check_dataset.py
For each photo it checks: image size matches the calibration, the ArUco reference
marker (ID 11) is found, the marker sits inside the safe zone (15% in from every
edge) and the camera tilt measured from the marker. A report is saved to
dataset/check_report.csv.
"""
import argparse
import csv
import glob
import math
import os

import cv2
import numpy as np

MARKER_ID = 11
MARKER_MM = 30.0   # replace with the side of the black square measured with your ruler
SAFE_MARGIN = 0.15  # marker must be at least 15% of the image away from every edge
MAX_TILT_DEG = 5.0  # photos tilted more than this are flagged
READ_FLAGS = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="dataset/raw")
    parser.add_argument("--params", default="calibration/camera_params.npz")
    parser.add_argument("--report", default="dataset/check_report.csv")
    args = parser.parse_args()

    params = np.load(args.params)
    camera_matrix, dist = params["camera_matrix"], params["dist_coeffs"]
    calib_size = tuple(int(v) for v in params["image_size"])

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_6X6_250), cv2.aruco.DetectorParameters()
    )
    half = MARKER_MM / 2.0
    marker_object = np.array(
        [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float32
    )

    paths = sorted(
        p for p in glob.glob(os.path.join(args.images, "*"))
        if p.lower().endswith((".jpg", ".jpeg"))
    )
    print(f"Checking {len(paths)} photos in {os.path.abspath(args.images)}\n")
    print(f"{'photo':<16}{'marker px':>10}{'tilt deg':>10}  status")

    rows = []
    for path in paths:
        name = os.path.basename(path)
        image = cv2.imread(path, READ_FLAGS)
        if image is None:
            rows.append((name, "", "", "UNREADABLE"))
            print(f"{name:<16}{'':>10}{'':>10}  UNREADABLE")
            continue
        h, w = image.shape[:2]
        problems = []
        side_px, tilt = "", ""
        if (w, h) != calib_size:
            problems.append(f"SIZE {w}x{h}")
        corners, ids, _ = detector.detectMarkers(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        index = None if ids is None else next(
            (i for i, v in enumerate(ids.flatten()) if v == MARKER_ID), None)
        if index is None:
            problems.append("NO MARKER")
        else:
            c = corners[index].reshape(4, 2)
            side_px = round(
                float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)])))
            inside = ((c[:, 0] > SAFE_MARGIN * w) & (c[:, 0] < (1 - SAFE_MARGIN) * w) &
                      (c[:, 1] > SAFE_MARGIN * h) & (c[:, 1] < (1 - SAFE_MARGIN) * h)).all()
            if not inside:
                problems.append("MARKER NEAR EDGE")
            ok, rvec, _ = cv2.solvePnP(marker_object, c, camera_matrix, dist,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                normal_z = cv2.Rodrigues(rvec)[0][2, 2]
                tilt = round(math.degrees(
                    math.acos(min(1.0, abs(float(normal_z))))), 1)
                if tilt > MAX_TILT_DEG:
                    problems.append("TILTED")
        status = ", ".join(problems) if problems else "ok"
        rows.append((name, side_px, tilt, status))
        print(f"{name:<16}{side_px!s:>10}{tilt!s:>10}  {status}")

    with open(args.report, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["photo", "marker_side_px", "tilt_deg", "status"])
        writer.writerows(rows)

    good = sum(1 for r in rows if r[3] == "ok")
    print(f"\n{good} of {len(rows)} photos are ok.")
    flagged = [r for r in rows if r[3] != "ok"]
    if flagged:
        print("Review or retake these:")
        for name, _, _, status in flagged:
            print(f"  {name}: {status}")
    print(f"Report saved to {args.report}")


if __name__ == "__main__":
    main()
