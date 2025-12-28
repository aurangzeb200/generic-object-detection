from canny_utils import canny_custom
import numpy as np

def grayscale_converion(image):
    B = image[:, :, 0].astype(np.float32)
    G = image[:, :, 1].astype(np.float32)
    R = image[:, :, 2].astype(np.float32)

    gray = 0.114 * B + 0.587 * G + 0.299 * R
    return gray.astype(np.uint8)

def canny_edge_detector(image, low_threshold=7, high_threshold=17):

    gray = grayscale_converion(image)
    return canny_custom(gray, sigma=1.0, Th=high_threshold, Tl=low_threshold)
