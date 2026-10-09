"""
ViT – Vision Transformer for image restoration.
Taken directly from the original notebook with minimal refactoring.
PatchEmbed + positional embedding + TransformerBlock stack + CNN decode head.
"""

import tensorflow as tf
from tensorflow import keras

layers = keras.layers


# ── Patch embedding ───────────────────────────────────────────────────────────

class PatchEmbed(layers.Layer):
    """
    Splits image (B, H, W, C) into non-overlapping patches and projects each
    to a vector of dimension `embed_dim`.  Uses a Conv2D with stride = patch_size
    (equivalent to per-patch linear projection).
    """
    def __init__(self, patch_size: int, embed_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.proj    = layers.Conv2D(embed_dim, kernel_size=patch_size,
                                     strides=patch_size, padding='valid')
        self.reshape = layers.Reshape((-1, embed_dim))

    def call(self, x):
        x = self.proj(x)      # (B, H/P, W/P, D)
        x = self.reshape(x)   # (B, N, D)
        return x


# ── Transformer block ─────────────────────────────────────────────────────────

class TransformerBlock(layers.Layer):
    """
    Standard ViT block:
        x → LayerNorm → MultiHeadAttention → residual
          → LayerNorm → MLP (Dense-GELU-Dense) → residual
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int,
                 dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn  = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp   = keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x, training: bool = False):
        x = x + self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x


# ── CNN decode head ───────────────────────────────────────────────────────────

def _build_cnn_head(tokens, grid_size: int, embed_dim: int, out_channels: int = 1):
    """
    Reshape token sequence back to spatial feature map, then upsample ×8
    with ConvTranspose2D layers to reach the original image resolution.
    """
    x = layers.Reshape((grid_size, grid_size, embed_dim))(tokens)  # (B, G, G, D)

    # ×2 → 2G
    x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)

    # ×2 → 4G
    x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)

    # ×2 → 8G  (= img_size when G = img_size // patch_size // 8 * 8)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding='same', use_bias=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(out_channels, 1, activation='sigmoid')(x)
    return x


# ── Full model ────────────────────────────────────────────────────────────────

def build_vit(
    img_size:   int   = 128,
    patch_size: int   = 8,
    embed_dim:  int   = 256,
    num_heads:  int   = 8,
    mlp_dim:    int   = 512,
    num_layers: int   = 6,
    dropout:    float = 0.1,
) -> keras.Model:
    """
    ViT restoration model.

    Args:
        img_size   : height = width (must be divisible by patch_size).
        patch_size : size of each patch (default 8 → 16×16 = 256 patches for 128×128).
        embed_dim  : token embedding dimension (must be divisible by num_heads).
        num_heads  : attention heads.
        mlp_dim    : hidden dim inside MLP block.
        num_layers : number of stacked transformer blocks.
        dropout    : dropout rate.
    """
    num_patches = (img_size // patch_size) ** 2
    grid_size   = img_size // patch_size

    inputs = layers.Input((img_size, img_size, 1))

    # (a) Patch embedding
    tokens = PatchEmbed(patch_size, embed_dim, name='patch_embed')(inputs)

    # (b) Learned positional embedding
    pos_embed = layers.Embedding(input_dim=num_patches, output_dim=embed_dim,
                                 name='pos_embed')
    positions = tf.range(num_patches)
    tokens    = tokens + pos_embed(positions)

    tokens = layers.Dropout(dropout)(tokens)

    # (c) Transformer blocks
    for i in range(num_layers):
        tokens = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout,
                                  name=f'transformer_block_{i}')(tokens)

    tokens = layers.LayerNormalization(epsilon=1e-6, name='final_norm')(tokens)

    # (d) CNN decode head
    outputs = _build_cnn_head(tokens, grid_size, embed_dim, out_channels=1)

    return keras.Model(inputs, outputs, name='ViT_Restoration')
