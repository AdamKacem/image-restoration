"""
Simple CNN baseline for image restoration.
Lightweight residual CNN — quick to train, useful as a baseline.
"""

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


def build_cnn(img_size: int = 128, filters: int = 64, depth: int = 6) -> keras.Model:
    """
    Residual CNN image-restoration baseline.

    Args:
        img_size : height = width of input images.
        filters  : number of convolutional filters per layer.
        depth    : number of intermediate residual blocks.
    """
    inputs = layers.Input((img_size, img_size, 1), name='input')

    # Initial feature extraction
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Residual body
    for i in range(depth):
        skip = x
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, skip])
        x = layers.Activation('relu')(x)

    # Output projection
    outputs = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    return keras.Model(inputs, outputs, name='CNN_Restoration')
