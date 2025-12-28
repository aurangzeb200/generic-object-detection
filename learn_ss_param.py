import cv2
import os
import numpy as np
from utils import compute_iou, read_gt_box, integral, rect_sum
from skimage.segmentation import felzenszwalb
from tqdm import tqdm

SCALES = [50, 75, 100, 150, 200]
BINS = 20
NUM_WINDOWS = 1000

def random_windows(w, h, num):
    x1 = np.random.randint(0, w-1, size=num)
    y1 = np.random.randint(0, h-1, size=num)
    x2 = np.array([np.random.randint(x1[i]+1, w) for i in range(num)])
    y2 = np.array([np.random.randint(y1[i]+1, h) for i in range(num)])
    return np.stack([x1, y1, x2, y2], axis=1)

best_scale = None
best_sep = -1

image_files = [f for f in os.listdir("data") if f.endswith(".jpg")]

for scale in SCALES:
    pos_hist = np.zeros(BINS)
    neg_hist = np.zeros(BINS)
    
    print(f"\nEvaluating scale: {scale}")
    
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join("data", img_name)
        xml_path = os.path.join("annotation", img_name.replace(".jpg", ".xml"))
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        gt = read_gt_box(xml_path)
        h_orig, w_orig = img.shape[:2]
        
        segments = felzenszwalb(img, scale=scale, sigma=0.5, min_size=50)
        unique_ids = np.unique(segments)
        ii_masks = {s_id: integral((segments==s_id).astype(float)) for s_id in unique_ids}
        s_area = {s_id: np.sum(segments==s_id) for s_id in unique_ids}
        
        windows = random_windows(w_orig, h_orig, NUM_WINDOWS)
        
        for win in windows:
            x1, y1, x2, y2 = win
            
            penalty = 0.0
            for s_id in unique_ids:
                area_in = rect_sum(ii_masks[s_id], x1, y1, x2, y2)
                if area_in > 0:
                    area_out = s_area[s_id] - area_in
                    penalty += min(area_in, area_out)
            score = 1 - (penalty / ((x2-x1+1)*(y2-y1+1) + 1e-8))
            
            iou = compute_iou(win, gt)
            
            bin_idx = min(int(score * BINS), BINS-1)
            if iou > 0.5:
                pos_hist[bin_idx] += 1
            else:
                neg_hist[bin_idx] += 1
        
        del segments, ii_masks, s_area

    pos_hist /= (np.sum(pos_hist) + 1e-8)
    neg_hist /= (np.sum(neg_hist) + 1e-8)
    
    separation = np.sum(pos_hist - neg_hist)
    if separation > best_sep:
        best_sep = separation
        best_scale = scale

print("\nBest SS scale:", best_scale)
