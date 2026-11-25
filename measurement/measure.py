import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

SCALE = 0.25  # Same scale used during training (4096x3072 -> 1024x768)
ARUCO_ID = 11
ARUCO_SIZE_MM = 30.0
GT_WIDTH_MM = 54.0
GT_HEIGHT_MM = 85.0

ARUCO_DICTS = [
    cv2.aruco.DICT_4X4_50,
    cv2.aruco.DICT_5X5_50,
    cv2.aruco.DICT_6X6_50,
    cv2.aruco.DICT_4X4_250,
    cv2.aruco.DICT_5X5_250,
    cv2.aruco.DICT_6X6_250,
    cv2.aruco.DICT_ARUCO_ORIGINAL,
]


def load_camera_params(npz_path="calibration/camera_params.npz"):
    data = np.load(npz_path)
    cam_mtx = data["camera_matrix"] if "camera_matrix" in data else data["mtx"]
    dist = data["dist_coeffs"] if "dist_coeffs" in data else data["dist"]
    return cam_mtx, dist


def load_model(weights_path="models/best_maskrcnn.pth", device="cpu"):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=None, weights_backbone=None
    )
    in_f = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 2)
    in_m = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_m, 256, 2)

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def load_image_landscape(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    h, w = img.shape[:2]
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def detect_aruco_scale(img_bgr, target_id=ARUCO_ID, marker_size_mm=ARUCO_SIZE_MM):
    gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMax = 53

    # Try full-res, 0.5x, and 0.25x scales so 12MP markers are never missed
    for scale in (1.0, 0.5, 0.25):
        if scale == 1.0:
            gray = gray_full
        else:
            gray = cv2.resize(gray_full, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        for dict_id in ARUCO_DICTS:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            detector = cv2.aruco.ArucoDetector(aruco_dict, params)
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is not None:
                flat_ids = ids.flatten().tolist()
                if target_id in flat_ids:
                    idx = flat_ids.index(target_id)
                    pts = corners[idx][0] / scale  # Map corners back to full-res pixels
                    sides = [
                        np.linalg.norm(pts[0] - pts[1]),
                        np.linalg.norm(pts[1] - pts[2]),
                        np.linalg.norm(pts[2] - pts[3]),
                        np.linalg.norm(pts[3] - pts[0]),
                    ]
                    avg_side_px = float(np.mean(sides))
                    mm_per_pixel = marker_size_mm / avg_side_px  # Pixel-to-mm conversion factor
                    return mm_per_pixel, pts, avg_side_px

    raise RuntimeError(f"ArUco marker ID {target_id} not detected.")


@torch.no_grad()
def predict_card_mask(model, img_bgr, device="cpu", score_thresh=0.5):
    h0, w0 = img_bgr.shape[:2]
    w, h = int(round(w0 * SCALE)), int(round(h0 * SCALE))
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(device) / 255.0

    out = model([tensor])[0]
    scores = out["scores"].cpu().numpy()
    if len(scores) == 0 or scores[0] < score_thresh:
        raise RuntimeError("No card detected above confidence threshold.")

    mask_small = out["masks"][0, 0].cpu().numpy()
    mask_full = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_LINEAR)
    mask_bin = (mask_full >= 0.5).astype(np.uint8)
    return mask_bin, float(scores[0])


def measure_image(img_path, model, cam_mtx, dist, apply_undistort=True, save_vis_path=None, device="cpu"):
    # 1. Read image and orient to landscape (4096x3072)
    img = load_image_landscape(img_path)

    # 2. Apply camera undistortion (skipped in --raw mode)
    if apply_undistort:
        img = cv2.undistort(img, cam_mtx, dist)

    # 3. Detect ArUco marker -> compute mm_per_pixel
    mm_per_pixel, aruco_pts, aruco_px = detect_aruco_scale(img)

    # 4. Predict card mask with Mask R-CNN -> fit rotated bounding box
    mask_bin, score = predict_card_mask(model, img, device=device)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = np.int32(cv2.boxPoints(rect))

    # 5. Convert pixel dimensions to millimeters
    short_px, long_px = sorted(rect[1])
    width_mm = short_px * mm_per_pixel
    height_mm = long_px * mm_per_pixel

    if save_vis_path:
        os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
        vis = img.copy()
        overlay = vis.copy()
        overlay[mask_bin > 0] = (0, 255, 0)
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        cv2.drawContours(vis, [box], 0, (0, 255, 255), 8)
        cv2.polylines(vis, [np.int32(aruco_pts)], True, (255, 0, 0), 8)

        mode_txt = "UNDISTORTED" if apply_undistort else "RAW"
        label1 = f"{mode_txt} | W: {width_mm:.2f} mm (GT {GT_WIDTH_MM:.1f}) | H: {height_mm:.2f} mm (GT {GT_HEIGHT_MM:.1f})"
        label2 = f"Scale: {mm_per_pixel:.4f} mm/px | ArUco: {aruco_px:.1f} px | Conf: {score:.3f}"
        cv2.putText(vis, label1, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 0, 0), 12)
        cv2.putText(vis, label1, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 255, 255), 5)
        cv2.putText(vis, label2, (60, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 10)
        cv2.putText(vis, label2, (60, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)

        vis_small = cv2.resize(vis, (1280, 960), interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_vis_path, vis_small)

    return {
        "file": os.path.basename(img_path),
        "mode": "undistorted" if apply_undistort else "raw",
        "width_mm": round(width_mm, 3),
        "height_mm": round(height_mm, 3),
        "w_err_mm": round(abs(width_mm - GT_WIDTH_MM), 3),
        "h_err_mm": round(abs(height_mm - GT_HEIGHT_MM), 3),
        "mm_per_pixel": round(mm_per_pixel, 5),
        "conf": round(score, 4),
    }


def run_eval(model, cam_mtx, dist, device="cpu"):
    with open("dataset/labels/test.json", "r", encoding="utf-8") as f:
        test_coco = json.load(f)

    test_files = [img["file_name"] for img in test_coco["images"]]
    rows = []
    failed = []

    print(f"\nEvaluating {len(test_files)} held-out test images (GT: {GT_WIDTH_MM} x {GT_HEIGHT_MM} mm)...\n")
    print(f"{'Image':<14} | {'Undist W/H (mm)':<19} | {'Undist Err':<12} | {'Raw W/H (mm)':<19} | {'Raw Err':<12}")
    print("-" * 88)

    u_errs, r_errs = [], []
    for fname in test_files:
        raw_path = os.path.join("dataset/raw", fname)
        vis_u = os.path.join("measurement/outputs", f"undist_{fname}")
        try:
            res_u = measure_image(raw_path, model, cam_mtx, dist, apply_undistort=True, save_vis_path=vis_u, device=device)
            res_r = measure_image(raw_path, model, cam_mtx, dist, apply_undistort=False, save_vis_path=None, device=device)
        except Exception as e:
            failed.append((fname, str(e)))
            print(f"{fname:<14} | FAILED: {e}")
            continue

        rows.extend([res_u, res_r])
        u_errs.extend([res_u["w_err_mm"], res_u["h_err_mm"]])
        r_errs.extend([res_r["w_err_mm"], res_r["h_err_mm"]])

        print(
            f"{fname:<14} | {res_u['width_mm']:>6.2f} x {res_u['height_mm']:<6.2f}   | "
            f"{res_u['w_err_mm']:>4.2f} / {res_u['h_err_mm']:<4.2f} | "
            f"{res_r['width_mm']:>6.2f} x {res_r['height_mm']:<6.2f}   | "
            f"{res_r['w_err_mm']:>4.2f} / {res_r['h_err_mm']:<4.2f}"
        )

    u_mae, u_max = float(np.mean(u_errs)), float(np.max(u_errs))
    r_mae, r_max = float(np.mean(r_errs)), float(np.max(r_errs))

    print("-" * 88)
    print(f"EVALUATED      -> {len(test_files) - len(failed)}/{len(test_files)} images")
    print(f"UNDISTORTED    -> MAE: {u_mae:.3f} mm | Max Error: {u_max:.3f} mm")
    print(f"RAW (No Undist)-> MAE: {r_mae:.3f} mm | Max Error: {r_max:.3f} mm")
    print(f"DIFFERENCE (Raw - Undistorted) -> {r_mae - u_mae:+.3f} mm MAE")

    os.makedirs("measurement", exist_ok=True)
    with open("measurement/eval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "n_test_images": len(test_files),
        "n_evaluated": len(test_files) - len(failed),
        "failed": failed,
        "ground_truth_mm": {"width": GT_WIDTH_MM, "height": GT_HEIGHT_MM},
        "undistorted": {"mae_mm": round(u_mae, 4), "max_err_mm": round(u_max, 4)},
        "raw": {"mae_mm": round(r_mae, 4), "max_err_mm": round(r_max, 4)},
        "mae_diff_raw_minus_undist_mm": round(r_mae - u_mae, 4),
    }
    with open("measurement/eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved: measurement/eval_results.csv, measurement/eval_summary.json, and visuals in measurement/outputs/")


def main():
    parser = argparse.ArgumentParser(description="Measure Metro Card using ArUco + Mask R-CNN")
    parser.add_argument("--image", type=str, help="Path to a single raw image to measure")
    parser.add_argument("--raw", action="store_true", help="Skip camera undistortion")
    parser.add_argument("--eval", action="store_true", help="Evaluate all 7 test images (Undistorted vs Raw)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cam_mtx, dist = load_camera_params()
    model = load_model(device=device)

    if args.eval:
        run_eval(model, cam_mtx, dist, device=device)
    elif args.image:
        out_name = os.path.basename(args.image)
        vis_path = os.path.join("measurement/outputs", f"demo_{out_name}")
        res = measure_image(
            args.image, model, cam_mtx, dist,
            apply_undistort=(not args.raw),
            save_vis_path=vis_path,
            device=device,
        )
        print(json.dumps(res, indent=2))
        print(f"Saved visualization: {vis_path}")
    else:
        print("Usage:\n  python measurement/measure.py --image dataset/raw/card_010.jpg\n  python measurement/measure.py --eval")
        sys.exit(1)


if __name__ == "__main__":
    main()
