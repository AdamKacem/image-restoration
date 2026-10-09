"""
Evaluation metrics used across the project.
- PSNR  (Peak Signal-to-Noise Ratio)
- SSIM  (Structural Similarity Index)
- MAE   (Mean Absolute Error)

All functions accept numpy arrays in [0, 1] with shape (N, H, W, 1).
"""

import numpy as np
import tensorflow as tf


def compute_psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PSNR in dB. Higher is better. Typical range: 25–40 dB."""
    mse = np.mean((y_true.astype(np.float32) - y_pred.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return float(10.0 * np.log10(1.0 / mse))


def compute_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean SSIM over all samples. Uses tf.image.ssim (range [0,1]).
    Higher is better.
    """
    t = tf.constant(y_true, dtype=tf.float32)
    p = tf.constant(y_pred, dtype=tf.float32)
    ssim_vals = tf.image.ssim(t, p, max_val=1.0)
    return float(tf.reduce_mean(ssim_vals).numpy())


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error. Lower is better."""
    return float(np.mean(np.abs(y_true.astype(np.float32) - y_pred.astype(np.float32))))


def compute_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict with all three metrics."""
    return {
        'psnr': compute_psnr(y_true, y_pred),
        'ssim': compute_ssim(y_true, y_pred),
        'mae':  compute_mae(y_true, y_pred),
    }
