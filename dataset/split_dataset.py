"""Split dataset/labels/annotations.json into train/val/test COCO JSONs
and save a 6-image visual sanity check (dataset/mask_preview.jpg).

Usage (from repo root):
    python dataset/split_dataset.py
"""
import json
import os

import cv2
import numpy as np

ANN_PATH = "dataset/labels/annotations.json"
IMG_DIR = "dataset/undistorted"
OUT_DIR = "dataset/labels"
PREVIEW_PATH = "dataset/mask_preview.jpg"

TEST_NAMES = {
    "card_010.jpg", "card_020.jpg", "card_030.jpg", "card_040.jpg",
    "card_050.jpg", "card_060.jpg", "card_070.jpg",
}
VAL_NAMES = {
    "card_003.jpg", "card_007.jpg", "card_013.jpg", "card_017.jpg",
    "card_023.jpg", "card_027.jpg", "card_033.jpg", "card_037.jpg",
    "card_043.jpg", "card_047.jpg", "card_053.jpg", "card_057.jpg",
    "card_063.jpg", "card_067.jpg",
}


def save_split(data, img_list, out_path):
    keep_ids = {im["id"] for im in img_list}
    anns = [a for a in data["annotations"] if a["image_id"] in keep_ids]
    out = {
        "info": data.get("info", {}),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": img_list,
        "annotations": anns,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return len(img_list), len(anns)


def make_preview(data):
    anns_by_img = {}
    for a in data["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    sample_names = [
        "card_001.jpg", "card_010.jpg", "card_027.jpg",
        "card_040.jpg", "card_057.jpg", "card_077.jpg",
    ]
    by_name = {im["file_name"]: im for im in data["images"]}
    tiles = []

    for name in sample_names:
        im_info = by_name.get(name)
        if im_info is None:
            continue
        img = cv2.imread(os.path.join(IMG_DIR, name))
        overlay = img.copy()
        for a in anns_by_img.get(im_info["id"], []):
            for poly in a["segmentation"]:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.polylines(img, [pts], True, (0, 255, 0), 8)
        blended = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        tile = cv2.resize(blended, (640, 480))
        tag = "TEST" if name in TEST_NAMES else ("VAL" if name in VAL_NAMES else "TRAIN")
        cv2.putText(tile, f"{name} [{tag}]", (15, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        tiles.append(tile)

    if len(tiles) == 6:
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
        cv2.imwrite(PREVIEW_PATH, grid)
        print(f"Saved visual check to {PREVIEW_PATH}")


def main():
    with open(ANN_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    train_imgs, val_imgs, test_imgs = [], [], []
    for im in data["images"]:
        name = im["file_name"]
        if name in TEST_NAMES:
            test_imgs.append(im)
        elif name in VAL_NAMES:
            val_imgs.append(im)
        else:
            train_imgs.append(im)

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
        n_img, n_ann = save_split(data, img_list, os.path.join(OUT_DIR, f"{split_name}.json"))
        print(f"{split_name:5s}: {n_img:2d} images, {n_ann:2d} annotations -> {OUT_DIR}/{split_name}.json")

    make_preview(data)


if __name__ == "__main__":
    main()
