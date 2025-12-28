import numpy as np
import cv2
from skimage.segmentation import felzenszwalb
from utils import integral, rect_sum
from edge_detection import canny_edge_detector

def spectral_residual(img_gray):
    f = np.fft.fft2(img_gray)
    amp = np.abs(f)
    phase = np.angle(f)
    log_amp = np.log(amp + 1e-8)
    avg_log_amp = cv2.blur(log_amp, (3, 3))
    residual = log_amp - avg_log_amp
    res_f = np.exp(residual + 1j * phase)
    saliency = np.abs(np.fft.ifft2(res_f)) ** 2
    return cv2.GaussianBlur(saliency, (9, 9), 2.5)

def ms_score(img, window, thresh, scale):
    x1, y1, x2, y2 = window
    h, w = img.shape[:2]
    total_density = 0.0
    for s in scale:
        resized = cv2.resize(img, (s, s))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        sal = spectral_residual(gray)
        sal = cv2.resize(sal, (w, h))
        sal = sal / (sal.max() + 1e-8)
        binary = (sal > thresh).astype(float)
        ii = integral(binary)
        density = rect_sum(ii, x1, y1, x2, y2) / (
            (x2 - x1 + 1) * (y2 - y1 + 1) + 1e-8
        )
        total_density += density
    return total_density / len(scale)

def ss_score(img, window, scale):
    x1, y1, x2, y2 = window
    area_w = (x2-x1+1)*(y2-y1+1)
    segments = felzenszwalb(img, scale=scale, sigma=0.5, min_size=50)
    penalty = 0.0
    for s_id in np.unique(segments):
        mask = (segments==s_id).astype(float)
        ii_mask = integral(mask)
        area_in = rect_sum(ii_mask, x1, y1, x2, y2)
        if area_in>0:
            area_out = np.sum(mask)-area_in
            penalty += min(area_in, area_out)
    return 1 - penalty/(area_w+1e-8)

def ed_score(img, window, theta_ed=0.2):
    x1, y1, x2, y2 = window
    w_width = x2 - x1
    w_height = y2 - y1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = canny_edge_detector(img, low_threshold=7, high_threshold=17) / 255.0
    ii_edges = integral(edges)
    pad_w = int(w_width*theta_ed)
    pad_h = int(w_height*theta_ed)
    full_sum = rect_sum(ii_edges,x1,y1,x2,y2)
    inner_sum = rect_sum(ii_edges,x1+pad_w,y1+pad_h,x2-pad_w,y2-pad_h)
    edges_in_ring = full_sum - inner_sum
    perimeter = 2*(w_width+w_height)
    return edges_in_ring/(perimeter+1e-8)
