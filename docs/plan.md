# Plan: Measure It (metro card width/height in mm)

Start time: (2026-11-23 12:15)

## Goal
Photo se PMA metro card ki width aur height mm mein batana, ArUco reference marker se scale nikaal kar.

## Decisions already made
- Object: PMA metro card. Reference: printed ArUco marker (DICT_6X6_250, ID 11), side ruler se naapi hui, card ke saath ek hi plane par.
- Camera: OnePlus Turbo 6, 1x main lens, focus/zoom/resolution locked, phone seedha upar se.
- Ground truth: ruler (resolution 1 mm, reading error ~±0.5 mm), 3-5 readings ka average.
- Calibration: ChArUco 5x7, 25 mm squares, OpenCV.
- Model: Mask R-CNN (torchvision, pretrained backbone), Colab T4 par train. Reason: instance segmentation, COCO metrics seedhe milte hain.
- Inference and timing: ThinkPad T495, Ryzen, CPU only.
- Split: 70/20/10, fixed list in dataset/splits.md. Test split training, tuning ya checkpoint choose karne mein kabhi use nahi hoga.

## Time budget (8 hours)
| Part | Time |
|---|---|
| Repo, plan, split | 0:15 |
| Calibration (20+ images, reprojection error) | 0:45 |
| Data collection (70+ images, undistort) | 1:00 |
| Labelling | 1:00 |
| Training (Colab, runs in background) | 1:30 |
| Inference command (image to mm) | 1:00 |
| Accuracy vs ruler (10+ instances) | 0:45 |
| Raw vs undistorted comparison | 0:30 |
| Docs, README, Loom | 1:00 |

## Priority
Items 1-4 (floor) pehle, phir 5-6. Items 7-10 sirf agar waqt bache.

## Already decided to skip
(abhi khali. Jab tay ho jaye ke kuch skip karna hai to yahan likho aur wajah likho.)

## Decision record (end mein bharna)
- Approach taken:
- Approach rejected:
- What rejecting it cost: