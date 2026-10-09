"""
Shared loss functions — extracted directly from the original codebase.
Each model can pick the loss that suits it best.
"""

import tensorflow as tf
from tensorflow import keras


# ── Haar DWT (needed by wavelet_loss) ────────────────────────────────────────

class _HaarDWT(keras.layers.Layer):
    """Single-level 2-D Haar discrete wavelet transform (for loss use only)."""
    def call(self, x):
        x00 = x[:, 0::2, 0::2, :]
        x01 = x[:, 0::2, 1::2, :]
        x10 = x[:, 1::2, 0::2, :]
        x11 = x[:, 1::2, 1::2, :]
        LL = (x00 + x01 + x10 + x11) * 0.5
        LH = (x00 + x01 - x10 - x11) * 0.5
        HL = (x00 - x01 + x10 - x11) * 0.5
        HH = (x00 - x01 - x10 + x11) * 0.5
        return tf.concat([LL, LH, HL, HH], axis=-1)


_dwt_loss = _HaarDWT(name='dwt_loss_global')


# ── Loss functions ────────────────────────────────────────────────────────────

def weighted_mse_ssim_loss(y_true, y_pred):
    """
    Used by: UNet, ViT.
    Dark pixels (ink) get high weight; light pixels (paper) get low weight.
    """
    weight    = 1.0 + 4.0 * (1.0 - y_true)
    mse_loss  = tf.reduce_mean(weight * tf.square(y_true - y_pred))
    ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    return 0.5 * mse_loss + 0.5 * ssim_loss

def make_weighted_mse_ssim(
    ink_weight:  float = 4.0,   # multiplier on dark pixels
    mse_alpha:   float = 0.5,   # weight of MSE term
    ssim_alpha:  float = 0.5,   # weight of SSIM term  (mse + ssim should = 1)
):
    """
    Returns a configured loss function.
    Use this instead of weighted_mse_ssim_loss when you want to tune coefficients.
    """
    def loss(y_true, y_pred):
        weight    = 1.0 + ink_weight * (1.0 - y_true)
        mse_loss  = tf.reduce_mean(weight * tf.square(y_true - y_pred))
        ssim_loss = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        return mse_alpha * mse_loss + ssim_alpha * ssim_loss
    loss.__name__ = f'wmse_ssim_iw{ink_weight}_a{mse_alpha}'
    return loss


def wavelet_loss(y_true, y_pred):
    """
    Used by: MWCNN.
    Multi-scale loss: weighted MSE + SSIM + LL-band MSE + HF-band MSE.
    """
    weight   = 1.0 + 4.0 * (1.0 - y_true)
    mse_full = tf.reduce_mean(weight * tf.square(y_true - y_pred))
    ssim_l   = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
    w_true   = _dwt_loss(y_true)
    w_pred   = _dwt_loss(y_pred)
    ll_loss  = tf.reduce_mean(tf.square(w_true[..., :1] - w_pred[..., :1])) * 2.0
    hf_loss  = tf.reduce_mean(tf.square(w_true[..., 1:] - w_pred[..., 1:])) * 1.0
    return 0.4 * mse_full + 0.3 * ssim_l + 0.2 * ll_loss + 0.1 * hf_loss


LOSS_MAP = {
    'weighted_mse_ssim': weighted_mse_ssim_loss,
    'wavelet':           wavelet_loss,
    'mse':               'mse',
    'mae':               'mae',
}
