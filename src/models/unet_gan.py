"""
UDBNet adversarial model (Pix2Pix-style GAN).
Taken directly from the original notebook with minimal refactoring:
- Generator  : U-Net with stride-2 downsampling + skip connections.
- Discriminator: PatchGAN.
- GAN wrapper : custom train_step (GAN + λ·L1).
- compile() signature is kept compatible with the trainer (see note below).

NOTE FOR TRAINER:
  Because the GAN uses its own compile(g_optimizer, d_optimizer), the trainer
  detects 'unet_gan' and calls gan.compile(...) with both optimizers.
  Prediction is exposed through gan(x) → generator(x).
"""

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


# ── Building blocks ───────────────────────────────────────────────────────────

def _down_block(x, filters: int, normalize: bool = True, dropout: float = 0.0):
    x = layers.Conv2D(filters, 4, strides=2, padding='same', use_bias=False)(x)
    if normalize:
        x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x


def _up_block(x, skip, filters: int, dropout: float = 0.0):
    x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    x = layers.Concatenate()([x, skip])
    return x


# ── Generator (adapted UDBNet, 128×128) ──────────────────────────────────────

def build_generator(img_size: int = 128) -> keras.Model:
    """
    7-level U-Net generator (128/2^7 = 1 → bottleneck).
    Output is in [-1, 1] (tanh).
    """
    inp = layers.Input((img_size, img_size, 1))

    d1 = _down_block(inp, 64,  normalize=False)           # 64×64
    d2 = _down_block(d1,  128)                            # 32×32
    d3 = _down_block(d2,  256)                            # 16×16
    d4 = _down_block(d3,  512, dropout=0.5)               #  8×8
    d5 = _down_block(d4,  512, dropout=0.5)               #  4×4
    d6 = _down_block(d5,  512, dropout=0.5)               #  2×2
    d7 = _down_block(d6,  512, normalize=False, dropout=0.5)  # 1×1 bottleneck

    u1 = _up_block(d7, d6, 512, dropout=0.5)             #  2×2
    u2 = _up_block(u1, d5, 512, dropout=0.5)             #  4×4
    u3 = _up_block(u2, d4, 512, dropout=0.5)             #  8×8
    u4 = _up_block(u3, d3, 256)                           # 16×16
    u5 = _up_block(u4, d2, 128)                           # 32×32
    u6 = _up_block(u5, d1,  64)                           # 64×64

    out = layers.Conv2DTranspose(1, 4, strides=2, padding='same')(u6)
    out = layers.Activation('tanh')(out)

    return keras.Model(inp, out, name='UDBNet_Generator')


# ── Discriminator (PatchGAN) ──────────────────────────────────────────────────

def build_discriminator(img_size: int = 128) -> keras.Model:
    """PatchGAN discriminator. Returns a map of logits (not a single scalar)."""
    inp = layers.Input((img_size, img_size, 1))

    x = layers.Conv2D(32,  4, strides=2, padding='same')(inp)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64,  4, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding='same')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(1, 4, padding='same')(x)   # logits
    return keras.Model(inp, x, name='UDBNet_Discriminator')


# ── GAN wrapper ───────────────────────────────────────────────────────────────

class UDBNetGAN(keras.Model):
    """
    Custom training loop: train discriminator → train generator.
    Loss = GAN (BCE) + lambda_l1 * L1.
    """

    def __init__(self, generator: keras.Model, discriminator: keras.Model,
                 lambda_l1: float = 100.0):
        super().__init__()
        self.generator     = generator
        self.discriminator = discriminator
        self.lambda_l1     = lambda_l1
        self.bce           = keras.losses.BinaryCrossentropy(from_logits=True)

    def compile(self, g_optimizer, d_optimizer):   # noqa: signature override
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def _gan_loss(self, pred, is_real: bool):
        target = tf.ones_like(pred) if is_real else tf.zeros_like(pred)
        return self.bce(target, pred)

    @tf.function
    def train_step(self, data):
        noisy, clean = data

        # 1. Discriminator
        with tf.GradientTape() as d_tape:
            fake   = self.generator(noisy, training=True)
            d_real = self.discriminator(clean, training=True)
            d_fake = self.discriminator(fake,  training=True)
            d_loss = (self._gan_loss(d_real, True) + self._gan_loss(d_fake, False)) * 0.5

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_grads, self.discriminator.trainable_variables))

        # 2. Generator
        with tf.GradientTape() as g_tape:
            fake       = self.generator(noisy, training=True)
            d_fake     = self.discriminator(fake, training=True)
            g_gan_loss = self._gan_loss(d_fake, True)
            g_l1_loss  = tf.reduce_mean(tf.abs(clean - fake))
            g_loss     = g_gan_loss + self.lambda_l1 * g_l1_loss

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_grads, self.generator.trainable_variables))

        return {'g_loss': g_loss, 'g_gan': g_gan_loss, 'g_l1': g_l1_loss, 'd_loss': d_loss}

    def call(self, x, training: bool = False):
        return self.generator(x, training=training)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_unet_gan(img_size: int = 128, lambda_l1: float = 100.0) -> UDBNetGAN:
    gen   = build_generator(img_size)
    disc  = build_discriminator(img_size)
    return UDBNetGAN(gen, disc, lambda_l1=lambda_l1)
