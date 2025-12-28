import cv2
import os
import numpy as np
from utils import compute_iou, read_gt_box, integral, rect_sum
from tqdm import tqdm
from canny_utils import canny_custom, grayscale_converion

THETAS = [0.05, 0.1, 0.15, 0.2, 0.25]
BINS = 20
NUM_WINDOWS = 1000

def random_window(w, h):
    x1 = np.random.randint(0, w-1)
    y1 = np.random.randint(0, h-1)
    x2 = np.random.randint(x1+1, w)
    y2 = np.random.randint(y1+1, h)
    return [x1, y1, x2, y2]

def ed_score_precomputed(ii_edges, window, theta_ed=0.2):
    x1, y1, x2, y2 = window
    w_width = x2 - x1
    w_height = y2 - y1

    pad_w = int(w_width * theta_ed)
    pad_h = int(w_height * theta_ed)

    full_sum = rect_sum(ii_edges, x1, y1, x2, y2)
    inner_sum = rect_sum(ii_edges, x1 + pad_w, y1 + pad_h, x2 - pad_w, y2 - pad_h)

    edges_in_ring = full_sum - inner_sum
    perimeter = 2 * (w_width + w_height)
    return edges_in_ring / (perimeter + 1e-8)


best_theta = None
best_sep = -1

image_files = [f for f in os.listdir("data") if f.endswith(".jpg")]

for theta in THETAS:
    pos, neg = [], []

    print(f"\nEvaluating theta: {theta}")
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join("data", img_name)
        xml_path = os.path.join("annotation", img_name.replace(".jpg", ".xml"))
        img = cv2.imread(img_path)
        gt = read_gt_box(xml_path)
        h, w = img.shape[:2]

        gray = grayscale_converion(img)
        edges = canny_custom(gray, sigma=1.0, Th=17, Tl=7) / 255.0  
        ii_edges = integral(edges)

        for _ in range(NUM_WINDOWS):
            win = random_window(w, h)
            score = ed_score_precomputed(ii_edges, win, theta)
            iou = compute_iou(win, gt)
            (pos if iou > 0.5 else neg).append(score)

    pos_hist, _ = np.histogram(pos, bins=BINS, range=(0,1), density=True)
    neg_hist, _ = np.histogram(neg, bins=BINS, range=(0,1), density=True)
    separation = np.sum(pos_hist - neg_hist)

    if separation > best_sep:
        best_sep = separation
        best_theta = theta

print("\nBest ED theta:", best_theta)
