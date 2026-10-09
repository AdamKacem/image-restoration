"""
MWCNN – Multi-level Wavelet Convolutional Neural Network.
Taken directly from the original notebook with minimal refactoring.
The Haar DWT/IWT layers and the corrected channel-alignment Conv1×1 are preserved.
"""

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


# ── Haar DWT / IWT ───────────────────────────────────────────────────────────

class HaarDWT(layers.Layer):
    """Single-level 2-D Haar discrete wavelet transform."""
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


class HaarIWT(layers.Layer):
    """Single-level 2-D Haar inverse wavelet transform."""
    def call(self, x):
        C  = tf.shape(x)[-1] // 4
        LL, LH = x[..., :C],       x[..., C:2*C]
        HL, HH = x[..., 2*C:3*C], x[..., 3*C:]
        x00 = (LL + LH + HL + HH) * 0.5
        x01 = (LL + LH - HL - HH) * 0.5
        x10 = (LL - LH + HL - HH) * 0.5
        x11 = (LL - LH - HL + HH) * 0.5
        B  = tf.shape(x)[0]
        H2 = tf.shape(x)[1]
        W2 = tf.shape(x)[2]
        row_top = tf.reshape(tf.stack([x00, x01], axis=3), [B, H2, W2 * 2, C])
        row_bot = tf.reshape(tf.stack([x10, x11], axis=3), [B, H2, W2 * 2, C])
        out = tf.reshape(tf.stack([row_top, row_bot], axis=2), [B, H2*2, W2*2, C])
        return out


# ── Residual CNN block ────────────────────────────────────────────────────────

def _cnn_block(x, filters: int, name_prefix: str = ''):
    skip = x
    if x.shape[-1] != filters:
        skip = layers.Conv2D(filters, 1, padding='same',
                             name=f'{name_prefix}_proj')(skip)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      name=f'{name_prefix}_c1')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Activation('relu', name=f'{name_prefix}_r1')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False,
                      name=f'{name_prefix}_c2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Add(name=f'{name_prefix}_add')([x, skip])
    x = layers.Activation('relu', name=f'{name_prefix}_r2')(x)
    return x


# ── Model ─────────────────────────────────────────────────────────────────────

def build_mwcnn(img_size: int = 128) -> keras.Model:
    """
    MWCNN with 3-level DWT/IWT.
    Works for any img_size that is a multiple of 8.

    NOTE: The channel-alignment Conv1×1 layers (align1, align2) fix the
    original Add-layer shape mismatch bug (preserved from original fix).
    """
    inputs = layers.Input((img_size, img_size, 1), name='input')

    dwt1 = HaarDWT(name='dwt1')
    dwt2 = HaarDWT(name='dwt2')
    dwt3 = HaarDWT(name='dwt3')
    iwt1 = HaarIWT(name='iwt1')
    iwt2 = HaarIWT(name='iwt2')
    iwt3 = HaarIWT(name='iwt3')

    # ── Encoder ──────────────────────────────────────────────────────────────
    w1 = dwt1(inputs)                        # (H/2, W/2, 4)
    w1 = _cnn_block(w1, 32, 'lvl1_dn')      # → 32

    w2 = dwt2(w1)                            # (H/4, W/4, 128)
    w2 = _cnn_block(w2, 64, 'lvl2_dn_a')
    w2 = _cnn_block(w2, 64, 'lvl2_dn_b')    # → 64

    w3 = dwt3(w2)                            # (H/8, W/8, 256)
    w3 = _cnn_block(w3, 128, 'lvl3_a')
    w3 = _cnn_block(w3, 128, 'lvl3_b')
    w3 = _cnn_block(w3, 128, 'lvl3_c')      # → 128

    # ── Decoder ──────────────────────────────────────────────────────────────
    w3_up = _cnn_block(w3, 128, 'lvl3_up')
    rec2  = iwt3(w3_up)                      # (H/4, W/4, 32)

    # FIX: align rec2 (32) → 64 before Add with w2 (64)
    rec2  = layers.Conv2D(64, 1, padding='same', name='align2')(rec2)
    w2_ref = layers.Add(name='add_lvl2')([rec2, w2])
    w2_ref = _cnn_block(w2_ref, 64, 'lvl2_up')

    rec1  = iwt2(w2_ref)                     # (H/2, W/2, 16)

    # FIX: align rec1 (16) → 32 before Add with w1 (32)
    rec1  = layers.Conv2D(32, 1, padding='same', name='align1')(rec1)
    w1_ref = layers.Add(name='add_lvl1')([rec1, w1])
    w1_ref = _cnn_block(w1_ref, 32, 'lvl1_up')

    out = iwt1(w1_ref)                       # (H, W, 8)
    out = layers.Conv2D(1, 3, padding='same', activation='sigmoid',
                        name='output_proj')(out)

    return keras.Model(inputs, out, name='MWCNN_Restoration')
