import numpy as np

# ==========================================
# 1. coordinate grid stuff
# ==========================================

def make_grid(height, width):
    # makes a grid of x,y coordinates. each cell holds its own (x, y) position
    y_vals = np.arange(height)
    x_vals = np.arange(width)
    gx, gy = np.meshgrid(x_vals, y_vals)
    grid = np.stack([gx, gy], axis=-1)
    return grid

def rotate_grid(coords, angle_deg, center=None):
    # rotate the whole grid around a center point.
    # numpy trig functions want radians, degrees need converting first
    ang = np.radians(angle_deg)
    c = np.cos(ang)
    s = np.sin(ang)

    # standard 2d rotation matrix
    R = np.array([[c, -s], [s, c]])

    H, W, _ = coords.shape
    if center is None:
        center = (W / 2.0, H / 2.0)

    # step 1: shift everything so the center sits at (0,0)
    shifted = coords.astype(np.float32) - np.array(center)
    # step 2: flatten into a list of points so matmul works easily
    flat = shifted.reshape(-1, 2)
    # step 3: rotate every point
    rotated = np.dot(flat, R.T)
    # step 4: shape back to a grid and move the center back
    rotated = rotated.reshape(H, W, 2) + np.array(center)
    return rotated

# ==========================================
# 2. color channels
# ==========================================

def split_channels(img):
    # separate red, green, blue into three 2d arrays.
    # did it with nested loops instead of slicing just to see it going on
    H, W, _ = img.shape
    R = np.zeros((H, W), dtype=img.dtype)
    G = np.zeros((H, W), dtype=img.dtype)
    B = np.zeros((H, W), dtype=img.dtype)
    for i in range(H):
        for j in range(W):
            R[i, j] = img[i, j, 0]
            G[i, j] = img[i, j, 1]
            B[i, j] = img[i, j, 2]
    return R, G, B

def brightness_contrast(img, alpha=1.2, beta=10):
    # new pixel = alpha * old pixel + beta
    f = img.astype(np.float32)
    out = alpha * f + beta
    out = np.clip(out, 0, 255)      # byte values only go 0..255
    return out.astype(np.uint8)

def rgb_to_hsv(img):
    # manual rgb -> hsv. returns hue(0-360), saturation(0-1), value(0-1)
    norm = img.astype(np.float32) / 255.0
    R = norm[:, :, 0]
    G = norm[:, :, 1]
    B = norm[:, :, 2]

    cmax = np.max(norm, axis=2)
    cmin = np.min(norm, axis=2)
    delta = cmax - cmin

    V = cmax

    # saturation is delta / max, but not when max is 0
    S = np.zeros_like(cmax)
    mask = cmax > 0
    S[mask] = delta[mask] / cmax[mask]

    # hue depends on which channel is the biggest
    H = np.zeros_like(cmax)

    # red biggest
    red_case = (cmax == R) & (delta != 0)
    H[red_case] = 60.0 * ((G[red_case] - B[red_case]) / delta[red_case]) + 360.0
    H[red_case] = np.mod(H[red_case], 360.0)

    # green biggest
    green_case = (cmax == G) & (delta != 0)
    H[green_case] = 60.0 * ((B[green_case] - R[green_case]) / delta[green_case]) + 120.0

    # blue biggest
    blue_case = (cmax == B) & (delta != 0)
    H[blue_case] = 60.0 * ((R[blue_case] - G[blue_case]) / delta[blue_case]) + 240.0

    return np.stack([H, S, V], axis=-1)

# ==========================================
# 3. blur / convolution
# ==========================================

def zero_pad(img, ph, pw):
    # put zeros around the border so the kernel has something to grab on the edges
    H, W = img.shape[:2]
    if img.ndim == 2:
        out = np.zeros((H + 2*ph, W + 2*pw), dtype=np.float32)
        out[ph:ph+H, pw:pw+W] = img
    else:
        C = img.shape[2]
        out = np.zeros((H + 2*ph, W + 2*pw, C), dtype=np.float32)
        out[ph:ph+H, pw:pw+W, :] = img
    return out

def convolve(img, kernel):
    # extremely basic convolution, just loops over every pixel
    # and every kernel cell. slow but it works
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = zero_pad(img, ph, pw)
    H, W = img.shape[:2]

    if img.ndim == 2:
        out = np.zeros((H, W), dtype=np.float32)
        for i in range(H):
            for j in range(W):
                total = 0.0
                for a in range(kh):
                    for b in range(kw):
                        total += padded[i + a, j + b] * kernel[a, b]
                out[i, j] = total
        return np.clip(out, 0, 255).astype(np.uint8)

    C = img.shape[2]
    out = np.zeros((H, W, C), dtype=np.float32)
    for ch in range(C):
        for i in range(H):
            for j in range(W):
                total = 0.0
                for a in range(kh):
                    for b in range(kw):
                        total += padded[i + a, j + b, ch] * kernel[a, b]
                out[i, j, ch] = total
    return np.clip(out, 0, 255).astype(np.uint8)

def gaussian_kernel(size=3, sigma=1.0):
    # builds a gaussian (bell curve) filter and normalizes it so the
    # weights all add up to 1 (image brightness stays the same)
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    return k / np.sum(k)

if __name__ == "__main__":
    print("=== python processor test ===\n")

    np.random.seed(42)
    img = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)

    grid = make_grid(8, 8)
    rotated = rotate_grid(grid, 45)

    hsv = rgb_to_hsv(img)
    adjusted = brightness_contrast(img, alpha=1.2, beta=10)

    kern = gaussian_kernel(size=3, sigma=1.0)
    blurred = convolve(img, kern)

    print("image shape:", img.shape)
    # checking that rotation actually moved the corner
    print("corner before rotation:", grid[0, 0])
    print("corner after rotation:", rotated[0, 0])
    print("hsv of top left pixel:", hsv[0, 0])
    print("blurred shape:", blurred.shape)