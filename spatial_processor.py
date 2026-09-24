import numpy as np

# ==========================================
# 1. 2D Coordinate Grid Transformations
# ==========================================

def create_coordinate_grid(height: int, width: int) -> np.ndarray:
    """
    Builds a 2D meshgrid of shape (H, W, 2) containing explicit (x, y) coordinates.
    """
    y_coords = np.arange(height)
    x_coords = np.arange(width)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    return np.stack([grid_x, grid_y], axis=-1)

def apply_2d_rotation(coords: np.ndarray, angle_degrees: float, center: tuple = None) -> np.ndarray:
    """
    Applies a 2D rotation matrix to coordinate grids around a center point without library helpers.
    """
    rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    
    # 2D Affine Rotation Matrix
    R = np.array([
        [cos_a, -sin_a],
        [sin_a,  cos_a]
    ])
    
    H, W, _ = coords.shape
    if center is None:
        center = (W / 2.0, H / 2.0)
        
    centered = coords - np.array(center)
    flat_coords = centered.reshape(-1, 2)
    rotated_flat = flat_coords @ R.T
    return rotated_flat.reshape(H, W, 2) + np.array(center)

# ==========================================
# 2. Pure NumPy Color Space Math
# ==========================================

def split_rgb_channels(img_rgb: np.ndarray):
    """
    Splits an (H, W, 3) RGB image array into discrete R, G, B channel 2D arrays via slicing.
    """
    return img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

def adjust_contrast_brightness(img: np.ndarray, alpha: float = 1.2, beta: float = 10.0) -> np.ndarray:
    """
    Applies linear intensity scaling: Output = alpha * Input + beta clamped to [0, 255].
    """
    img_float = img.astype(np.float32)
    adjusted = np.clip(alpha * img_float + beta, 0, 255)
    return adjusted.astype(np.uint8)

def rgb_to_hsv_numpy(img_rgb: np.ndarray) -> np.ndarray:
    """
    Converts RGB image arrays to HSV float space using pure NumPy vectorization.
    Outputs: Hue [0, 360), Saturation [0, 1], Value [0, 1].
    """
    norm_img = img_rgb.astype(np.float32) / 255.0
    R, G, B = norm_img[:, :, 0], norm_img[:, :, 1], norm_img[:, :, 2]
    
    c_max = np.max(norm_img, axis=-1)
    c_min = np.min(norm_img, axis=-1)
    delta = c_max - c_min
    
    # Value (V)
    V = c_max
    
    # Saturation (S)
    S = np.zeros_like(c_max)
    mask_max = c_max > 0
    S[mask_max] = delta[mask_max] / c_max[mask_max]
    
    # Hue (H)
    H = np.zeros_like(c_max)
    mask_r = (c_max == R) & (delta != 0)
    H[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / delta[mask_r]) + 360.0) % 360.0
    
    mask_g = (c_max == G) & (delta != 0)
    H[mask_g] = 60.0 * ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 120.0
    
    mask_b = (c_max == B) & (delta != 0)
    H[mask_b] = 60.0 * ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 240.0
    
    return np.stack([H, S, V], axis=-1)

# ==========================================
# 3. Spatial Blurring & 2D Convolution Engine
# ==========================================

def pad_array_2d(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """
    Manually pads 2D or 3D arrays with zeros along outer boundaries.
    """
    if img.ndim == 2:
        H, W = img.shape
        padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w), dtype=np.float32)
        padded[pad_h:pad_h + H, pad_w:pad_w + W] = img
    elif img.ndim == 3:
        H, W, C = img.shape
        padded = np.zeros((H + 2 * pad_h, W + 2 * pad_w, C), dtype=np.float32)
        padded[pad_h:pad_h + H, pad_w:pad_w + W, :] = img
    else:
        raise ValueError("Input array must be 2D or 3D.")
    return padded

def convolve2d_numpy(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Performs spatial 2D convolution using sliding window views and stride mechanics.
    """
    k_h, k_w = kernel.shape
    pad_h, pad_w = k_h // 2, k_w // 2
    
    img_float = img.astype(np.float32)
    padded = pad_array_2d(img_float, pad_h, pad_w)
    
    if img.ndim == 2:
        windows = np.lib.stride_tricks.sliding_window_view(padded, (k_h, k_w))
        output = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
    elif img.ndim == 3:
        H, W, C = img.shape
        output = np.zeros((H, W, C), dtype=np.float32)
        for c in range(C):
            windows = np.lib.stride_tricks.sliding_window_view(padded[:, :, c], (k_h, k_w))
            output[:, :, c] = np.tensordot(windows, kernel, axes=((2, 3), (0, 1)))
            
    return np.clip(output, 0, 255).astype(np.uint8)

def get_gaussian_kernel(size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Generates a normalized 2D Gaussian filter kernel.
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)

if __name__ == "__main__":
    print("=== Python Matrix Processor Verification ===")
    np.random.seed(42)
    sample_img = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)
    
    grid = create_coordinate_grid(8, 8)
    rotated_grid = apply_2d_rotation(grid, 45)
    
    hsv_img = rgb_to_hsv_numpy(sample_img)
    adjusted = adjust_contrast_brightness(sample_img, alpha=1.2, beta=10)
    
    gauss_kernel = get_gaussian_kernel(size=3, sigma=1.0)
    blurred = convolve2d_numpy(sample_img, gauss_kernel)
    
    print(f"Original Shape: {sample_img.shape}")
    print(f"Rotated Center Coord (4,4): {rotated_grid[4, 4]}")
    print(f"HSV Converted Pixel (0,0): {hsv_img[0, 0]}")
    print(f"Blurred Matrix Shape: {blurred.shape}")