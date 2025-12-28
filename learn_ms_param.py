import cv2
import os
import numpy as np
from cues import spectral_residual
from utils import compute_iou, read_gt_box

SCALES = [16, 24, 32, 48, 64]
THRESHOLDS = np.linspace(0.1, 1.0, 25)

best_t = None
best_scale = None
best_total_iou = 0.0

for scale in SCALES:
    for t in THRESHOLDS:
        total_iou = 0.0

        for img_name in os.listdir("data"):
            if not img_name.lower().endswith(".jpg"):
                continue

            img_path = os.path.join("data", img_name)
            xml_path = os.path.join("annotation", img_name.replace(".jpg", ".xml"))

            img = cv2.imread(img_path)
            if img is None:
                continue

            gt_box = read_gt_box(xml_path)
            h, w = img.shape[:2]

            resized = cv2.resize(img, (scale, scale))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            sal = spectral_residual(gray)
            sal = cv2.resize(sal, (w, h))
            sal = sal / (sal.max() + 1e-8)

            binary_map = (sal > t).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary_map)

            max_iou_img = 0.0
            for k in range(1, num_labels):
                ys, xs = np.where(labels == k)
                if len(xs) == 0:
                    continue
                blob_box = [xs.min(), ys.min(), xs.max(), ys.max()]
                max_iou_img = max(max_iou_img, compute_iou(blob_box, gt_box))

            total_iou += max_iou_img

        if total_iou > best_total_iou:
            best_total_iou = total_iou
            best_t = t
            best_scale = scale

print(f"Best Threshold: {best_t}")
print(f"Best Scale: {best_scale}")
print(f"Total IoU over training set: {best_total_iou:.4f}")
