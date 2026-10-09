"""
U-Net image restoration model.
Taken directly from the original notebook with minimal refactoring:
- build_unet() is extracted as a standalone function (no compile inside).
- compile() is handled by the trainer so lr / optimizer are fully CLI-controlled.
"""

import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models


def _conv_block(x, filters: int, dropout_rate: float = 0.0):
    """Two Conv2D + BN + ReLU, optional Dropout."""
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate)(x)
    return x


def build_unet(
    img_size:     int   = 128,
    dropout_rate: float = 0.3,
    base_filters: int   = 32,
) -> keras.Model:
    """
    Standard U-Net for image restoration.

    Args:
        img_size     : height = width of input.
        dropout_rate : dropout in encoder depth levels.
        base_filters : filters at the first encoder level (doubles each level).
    """
    f = base_filters
    inputs = layers.Input((img_size, img_size, 1))

    # ── Encoder ──────────────────────────────────────────────────────────────
    c1 = _conv_block(inputs, f)
    p1 = layers.MaxPooling2D(2)(c1)

    c2 = _conv_block(p1, f * 2)
    p2 = layers.MaxPooling2D(2)(c2)

    c3 = _conv_block(p2, f * 4)
    p3 = layers.MaxPooling2D(2)(c3)

    c4 = _conv_block(p3, f * 8, dropout_rate)
    p4 = layers.MaxPooling2D(2)(c4)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    c5 = _conv_block(p4, f * 16, dropout_rate)

    # ── Decoder ─────────────────────────────────────────────────────────────
    u6 = layers.Conv2DTranspose(f * 8, 2, strides=2, padding='same')(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = _conv_block(u6, f * 8, dropout_rate)

    u7 = layers.Conv2DTranspose(f * 4, 2, strides=2, padding='same')(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = _conv_block(u7, f * 4)

    u8 = layers.Conv2DTranspose(f * 2, 2, strides=2, padding='same')(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = _conv_block(u8, f * 2)

    u9 = layers.Conv2DTranspose(f, 2, strides=2, padding='same')(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = _conv_block(u9, f)

    # ── Output ───────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)

    return models.Model(inputs, outputs, name='UNet_Restoration')
