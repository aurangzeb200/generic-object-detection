import numpy as np

def normalize_to_u8(arr):
    """Normalizes any array to 0-255 uint8 range."""
    arr_float = arr.astype(np.float64)
    min_val, max_val = arr_float.min(), arr_float.max()
    if max_val - min_val == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr_float - min_val) / (max_val - min_val)
    return (norm * 255.0).astype(np.uint8)

def calculate_filter_size(sigma, T=0.3):
    sHalf = int(np.ceil(3.0 * sigma)) 
    N = 2 * sHalf + 1
    return N, sHalf

def calculate_gradient(N, sigma, scale_factor=255):
    sHalf = N // 2
    x = np.arange(-sHalf, sHalf + 1)
    y = np.arange(-sHalf, sHalf + 1)
    X, Y = np.meshgrid(x, y)
    
    G = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    
    Gx = -(X / sigma**2) * G
    Gy = -(Y / sigma**2) * G
    
    Gx_int = np.round(Gx * scale_factor).astype(np.int32)
    Gy_int = np.round(Gy * scale_factor).astype(np.int32)
    
    return Gx_int, Gy_int, scale_factor

def convolve(image, kernel):
    img_h, img_w = image.shape
    k_h, k_w = kernel.shape
    pad = k_h // 2
    
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode='edge').astype(np.float64)
    output = np.zeros_like(image, dtype=np.float64)
    
    kernel = np.flipud(np.fliplr(kernel))
    
    for i in range(img_h):
        for j in range(img_w):
            region = padded[i:i+k_h, j:j+k_w]
            output[i, j] = np.sum(region * kernel)
            
    return output

def apply_masks(image, Gx, Gy, scale_factor):
    fx = convolve(image, Gx)
    fy = convolve(image, Gy)
    
    fx = fx / scale_factor
    fy = fy / scale_factor
    
    return fx, fy

def compute_magnitude(fx, fy):
    M = np.hypot(fx, fy)
    M_u8 = normalize_to_u8(M)
    return M, M_u8

def compute_gradient_direction(fx, fy):
    phi = np.arctan2(fy, fx)
    phi_deg = np.degrees(phi) % 360
    
    phi_img = (phi_deg / 360.0 * 255).astype(np.uint8)
    return phi_img, phi_deg

def quantize_gradient_direction(phi_deg):
    q = np.zeros_like(phi_deg, dtype=np.uint8)
    
    mask0 = np.logical_or(
        np.logical_and(phi_deg >= 337.5, phi_deg < 360),
        np.logical_and(phi_deg >= 0, phi_deg < 22.5),
    )
    mask0 = np.logical_or(mask0, np.logical_and(phi_deg >= 157.5, phi_deg < 202.5))
    
    mask1 = np.logical_or(
        np.logical_and(phi_deg >= 22.5, phi_deg < 67.5),
        np.logical_and(phi_deg >= 202.5, phi_deg < 247.5)
    )
    
    mask2 = np.logical_or(
        np.logical_and(phi_deg >= 67.5, phi_deg < 112.5),
        np.logical_and(phi_deg >= 247.5, phi_deg < 292.5)
    )
    
    mask3 = np.logical_or(
        np.logical_and(phi_deg >= 112.5, phi_deg < 157.5),
        np.logical_and(phi_deg >= 292.5, phi_deg < 337.5)
    )
    
    q[mask0] = 0
    q[mask1] = 1
    q[mask2] = 2
    q[mask3] = 3
    
    return q

def non_maxima_suppression(M, q):
    rows, cols = M.shape
    Z = np.zeros((rows, cols), dtype=np.float64)
    
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle_code = q[i, j]
            
            try:
                q_val = 255
                r_val = 255
                
                if angle_code == 0:
                    q_val = M[i, j+1]
                    r_val = M[i, j-1]
                elif angle_code == 1:
                    q_val = M[i-1, j+1]
                    r_val = M[i+1, j-1]
                elif angle_code == 2:
                    q_val = M[i-1, j]
                    r_val = M[i+1, j]
                elif angle_code == 3:
                    q_val = M[i-1, j-1]
                    r_val = M[i+1, j+1]
                
                if (M[i, j] >= q_val) and (M[i, j] >= r_val):
                    Z[i, j] = M[i, j]
                else:
                    Z[i, j] = 0
                    
            except IndexError:
                pass
                
    return Z, normalize_to_u8(Z)

def hysteresis_thresholding(img, high, low):
    rows, cols = img.shape
    res = np.zeros((rows, cols), dtype=np.uint8)
    
    weak = 25
    strong = 255
    
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    stack = list(zip(strong_i, strong_j))
    
    while stack:
        i, j = stack.pop()
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                
                ni, nj = i + dx, j + dy
                if 0 <= ni < rows and 0 <= nj < cols:
                    if res[ni, nj] == weak:
                        res[ni, nj] = strong
                        stack.append((ni, nj))
    
    res[res != strong] = 0
    
    return res

def canny_custom(gray, sigma=1.0, Th=100, Tl=50):
    T_for_size = 0.3
    scale_factor = 255 

    N, sHalf = calculate_filter_size(sigma, T_for_size)
    
    Gx_int, Gy_int, sf = calculate_gradient(N, sigma, scale_factor)

    fx, fy = apply_masks(gray, Gx_int, Gy_int, sf)

    M_float, _ = compute_magnitude(fx, fy)

    phi_img, phi_deg = compute_gradient_direction(fx, fy)
    q = quantize_gradient_direction(phi_deg)

    suppressed_float, _ = non_maxima_suppression(M_float, q)

    edges = hysteresis_thresholding(suppressed_float, Th, Tl)

    return edges