"""
Synthetic data generation — taken directly from the original notebook.
Functions are unchanged; only the CLI-facing generate() wrapper is new.

Corruption types supported:
  - stamp   : elliptical stamp overlay (always applied)
  - blur    : Gaussian blur on clean image before stamp
  - noise   : Gaussian pixel noise (always applied)
  - occlusion : random black rectangle masking part of the image
"""

import os
import numpy as np
import cv2


# ═════════════════════════════════════════════════════════════════════════════
# Core generation helpers — UNCHANGED from original
# ═════════════════════════════════════════════════════════════════════════════

def add_paper_background(img: np.ndarray, img_size: int) -> np.ndarray:
    base       = np.random.uniform(0.82, 0.92)
    background = np.full((img_size, img_size), base, dtype=np.float32)

    large_noise = cv2.GaussianBlur(
        np.random.randn(img_size, img_size).astype(np.float32), (51, 51), 20
    ) * 0.04
    small_noise = cv2.GaussianBlur(
        np.random.randn(img_size, img_size).astype(np.float32), (5, 5), 2
    ) * 0.015
    background = np.clip(background + large_noise + small_noise, 0.0, 1.0)

    ink_color          = np.random.uniform(0.05, 0.25)
    result             = background.copy()
    result[img > 0.5]  = ink_color
    return result


def draw_digits(img_size: int) -> np.ndarray:
    canvas = np.zeros((img_size, img_size), dtype=np.float32)
    fonts  = [
        cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_PLAIN,   cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    ]
    num_digits = np.random.randint(1, 5)
    margin     = 12
    placed     = []

    for _ in range(num_digits):
        digit     = str(np.random.randint(0, 10))
        font      = fonts[np.random.randint(len(fonts))]
        scale     = np.random.uniform(0.9, 2.2)
        thickness = np.random.randint(1, 4)

        (tw, th), baseline = cv2.getTextSize(digit, font, scale, thickness)
        max_x = max(margin, img_size - tw - margin)
        max_y = max(th + margin, img_size - baseline - margin)
        if max_x <= margin or max_y <= th + margin:
            continue

        placed_ok = False
        for _ in range(20):
            x   = np.random.randint(margin, max_x)
            y   = np.random.randint(th + margin, max_y)
            pad = 4
            overlap = any(
                x < px + ptw + pad and x + tw + pad > px and
                y - th < py + pad  and y + pad > py - pth
                for (px, py, ptw, pth) in placed
            )
            if not overlap:
                placed_ok = True
                break

        if not placed_ok:
            continue

        canvas_u8 = (canvas * 255).astype(np.uint8)
        cv2.putText(canvas_u8, digit, (x, y), font, scale,
                    color=255, thickness=thickness, lineType=cv2.LINE_AA)
        canvas = canvas_u8.astype(np.float32) / 255.0
        placed.append((x, y - th, tw, th))

    return canvas


def add_stamp(img: np.ndarray, img_size: int) -> np.ndarray:
    stamp_layer = np.zeros((img_size, img_size), dtype=np.float32)
    cx  = np.random.randint(img_size // 4, 3 * img_size // 4)
    cy  = np.random.randint(img_size // 4, 3 * img_size // 4)
    rx  = np.random.randint(img_size // 6, img_size // 3)
    ry  = np.random.randint(img_size // 6, img_size // 3)
    ang = np.random.randint(0, 180)

    canvas_u8 = (stamp_layer * 255).astype(np.uint8)
    cv2.ellipse(canvas_u8, (cx, cy), (rx, ry), ang, 0, 360, color=200, thickness=2)
    inner_rx = max(4, rx - 6)
    inner_ry = max(4, ry - 6)
    cv2.ellipse(canvas_u8, (cx, cy), (inner_rx, inner_ry), ang, 0, 360,
                color=180, thickness=1)

    for i in range(np.random.randint(2, 6)):
        lx1 = int(np.clip(cx - inner_rx + np.random.randint(0, 10), 0, img_size - 1))
        lx2 = int(np.clip(cx + inner_rx - np.random.randint(0, 10), 0, img_size - 1))
        ly  = int(np.clip(cy - inner_ry // 2 + i * (inner_ry // max(1, 3)), 0, img_size - 1))
        cv2.line(canvas_u8, (lx1, ly), (lx2, ly),
                 color=int(np.random.randint(150, 210)), thickness=1)

    stamp_layer = cv2.GaussianBlur(canvas_u8.astype(np.float32) / 255.0, (5, 5), 1.5)
    opacity     = np.random.uniform(0.25, 0.6)
    stamp_val   = np.random.uniform(0.35, 0.60)
    blended     = img.copy()
    blended[stamp_layer > 0.1] = (
        (1 - opacity) * img[stamp_layer > 0.1] + opacity * stamp_val
    )
    return blended


def add_gaussian_noise(img: np.ndarray, img_size: int,
                        noise_level: float = 0.06) -> np.ndarray:
    std   = np.random.uniform(0.01, max(0.01, noise_level))
    noise = std * np.random.randn(img_size, img_size).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def add_blur(img: np.ndarray) -> np.ndarray:
    ksize = np.random.choice([3, 5, 7])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def add_occlusion(img: np.ndarray, img_size: int) -> np.ndarray:
    """Random black rectangle covering 10–25% of the image."""
    out = img.copy()
    w   = np.random.randint(img_size // 6, img_size // 3)
    h   = np.random.randint(img_size // 6, img_size // 3)
    x   = np.random.randint(0, img_size - w)
    y   = np.random.randint(0, img_size - h)
    out[y:y+h, x:x+w] = np.random.uniform(0.82, 0.92)  # paper-colored occlusion
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Main generation function
# ═════════════════════════════════════════════════════════════════════════════

def generate_data(
    num_samples:   int   = 1000,
    img_size:      int   = 128,
    noise_level:   float = 0.06,
    corruption:    str   = 'stamp',   # 'stamp' | 'blur' | 'occlusion' | 'all'
):
    """
    Generate paired (noisy, clean) image arrays.

    Returns:
        X_noisy : np.ndarray  shape (N, img_size, img_size, 1)  range [0,1]
        X_clean : np.ndarray  shape (N, img_size, img_size, 1)  range [0,1]
    """
    X_clean_list = []
    X_noisy_list = []

    for _ in range(num_samples):
        digit_mask = draw_digits(img_size)
        clean      = add_paper_background(digit_mask, img_size)
        noisy      = clean.copy()

        if corruption in ('stamp', 'all'):
            noisy = add_stamp(noisy, img_size)
        if corruption in ('blur', 'all'):
            noisy = add_blur(noisy)
        if corruption in ('occlusion', 'all'):
            noisy = add_occlusion(noisy, img_size)

        noisy = add_gaussian_noise(noisy, img_size, noise_level)

        X_clean_list.append(clean)
        X_noisy_list.append(noisy)

    X_noisy = np.array(X_noisy_list)[..., None]
    X_clean = np.array(X_clean_list)[..., None]
    return X_noisy, X_clean


def save_data(X_noisy: np.ndarray, X_clean: np.ndarray, output_dir: str) -> None:
    """Save arrays to output_dir as .npy files."""
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'X_noisy.npy'), X_noisy)
    np.save(os.path.join(output_dir, 'X_clean.npy'), X_clean)
    print(f"Saved {len(X_noisy)} samples to {output_dir}/")
    print(f"  X_noisy : {X_noisy.shape}  X_clean : {X_clean.shape}")


def load_data(data_dir: str):
    """Load noisy/clean arrays from data_dir."""
    noisy_path = os.path.join(data_dir, 'X_noisy.npy')
    clean_path = os.path.join(data_dir, 'X_clean.npy')
    if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
        raise FileNotFoundError(
            f"Data not found in {data_dir}. "
            "Run: python run.py generate --output <dir> first."
        )
    return np.load(noisy_path), np.load(clean_path)
