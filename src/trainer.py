"""
Trainer — handles the full training loop for all models.

Responsibilities:
  1. Build model from registry.
  2. Compile with chosen optimizer + loss.
  3. Set up Keras callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
  4. Run fit().
  5. Save weights, config, metrics, visual comparisons.

GAN models (unet_gan) have a custom train_step so they get special treatment.
"""

import os
import json
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')               # no display needed
import matplotlib.pyplot as plt

from .models       import get_model, AVAILABLE_MODELS
from .losses       import LOSS_MAP, weighted_mse_ssim_loss, wavelet_loss
from .metrics      import compute_all
from .data_generator import load_data


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_optimizer(name: str, lr: float):
    name = name.lower()
    if name == 'adam':
        return keras.optimizers.Adam(lr)
    if name == 'adamw':
        return keras.optimizers.AdamW(lr)
    if name == 'sgd':
        return keras.optimizers.SGD(lr, momentum=0.9)
    if name == 'rmsprop':
        return keras.optimizers.RMSprop(lr)
    raise ValueError(f"Unknown optimizer '{name}'. Choose: adam, adamw, sgd, rmsprop.")


def _default_loss(model_name: str):
    """Return the loss function that matches each model's original paper."""
    if model_name == 'mwcnn':
        return wavelet_loss
    return weighted_mse_ssim_loss


def _save_comparisons(X_noisy, X_clean, predictions, out_dir: str, n: int = 4) -> None:
    """Save [input | ground truth | prediction] comparison grid."""
    os.makedirs(out_dir, exist_ok=True)
    n = min(n, len(X_noisy))

    fig, axes = plt.subplots(3, n, figsize=(4 * n, 10))
    titles = ['Noisy (input)', 'Clean (target)', 'Prediction']
    sources = [X_noisy, X_clean, predictions]

    for row, (title, src) in enumerate(zip(titles, sources)):
        for col in range(n):
            axes[row, col].imshow(src[col, :, :, 0], cmap='gray', vmin=0, vmax=1)
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=11)
            axes[row, col].axis('off')

    plt.suptitle('Image Restoration Results', fontsize=13)
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'comparison.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    return save_path


def _save_loss_curve(history_dict: dict, out_dir: str) -> None:
    """Save training/validation loss curves."""
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(8, 4))
    for key, vals in history_dict.items():
        plt.plot(vals, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Metric')
    plt.title('Training history')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'loss_curve.png'), dpi=100)
    plt.close()


def _setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger('trainer')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s',
                            datefmt='%H:%M:%S')
    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # file
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# ── Main train function ───────────────────────────────────────────────────────

def train(args) -> None:
    """
    Full training pipeline driven by parsed CLI args.
    Saves everything under experiments/<exp_name>/.
    """
    exp_dir      = os.path.join('experiments', args.exp_name)
    weights_dir  = os.path.join(exp_dir, 'weights')
    outputs_dir  = os.path.join(exp_dir, 'outputs', 'comparisons')
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    log = _setup_logger(os.path.join(exp_dir, 'train.log'))
    log.info(f"Experiment : {args.exp_name}")
    log.info(f"Model      : {args.model}")

    # ── Save config ───────────────────────────────────────────────────────────
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    log.info(f"Config saved to {exp_dir}/config.json")

    # ── Load data ─────────────────────────────────────────────────────────────
    log.info(f"Loading data from {args.dataset} …")
    X_noisy, X_clean = load_data(args.dataset)
    log.info(f"  X_noisy {X_noisy.shape}  X_clean {X_clean.shape}")

    # ── Build model ───────────────────────────────────────────────────────────
    model_kwargs = dict(
        img_size     = args.img_size,
        dropout_rate = args.dropout,
        base_filters = args.base_filters,
        # vit params
        patch_size   = args.patch_size,
        embed_dim    = args.embed_dim,
        num_heads    = args.num_heads,
        mlp_dim      = args.mlp_dim,
        num_layers   = args.num_layers,
        # mwcnn has no extra params beyond img_size
    )
    log.info("Building model …")
    model = get_model(args.model, **model_kwargs)

    # ── Compile ───────────────────────────────────────────────────────────────
    opt  = _get_optimizer(args.optimizer, args.lr)
    from .losses import make_weighted_mse_ssim

    if args.loss in ('weighted_mse_ssim', 'auto') and (
        getattr(args, 'ink_weight',  None) is not None or
        getattr(args, 'mse_alpha',   None) is not None
    ):
        ink_weight = getattr(args, 'ink_weight', None)
        mse_alpha  = getattr(args, 'mse_alpha',  None)
        ink_weight = 4.0 if ink_weight is None else ink_weight
        mse_alpha  = 0.5 if mse_alpha  is None else mse_alpha
        # The two weights should add up to 1, so the SSIM weight follows the MSE weight.
        loss = make_weighted_mse_ssim(
            ink_weight = ink_weight,
            mse_alpha  = mse_alpha,
            ssim_alpha = 1.0 - mse_alpha,
        )
    else:
        loss = LOSS_MAP.get(args.loss, None) or _default_loss(args.model)

    is_gan = (args.model == 'unet_gan')

    if is_gan:
        # GAN needs two optimizers; use same lr for both by default
        opt2 = _get_optimizer(args.optimizer, args.lr)
        model.compile(g_optimizer=opt, d_optimizer=opt2)
        log.info("GAN model compiled (custom train_step).")
    else:
        model.compile(optimizer=opt, loss=loss, metrics=['mae'])
        log.info(f"Model compiled  loss={args.loss}  optimizer={args.optimizer}  lr={args.lr}")

    model.summary(print_fn=lambda s: log.info(s))

    # ── Callbacks ─────────────────────────────────────────────────────────────
    best_weights_path = os.path.join(weights_dir, 'best_weights.weights.h5')

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss' if not is_gan else 'g_loss',
            patience=args.patience,
            restore_best_weights=not is_gan,  # not supported for custom train_step
            verbose=1,
            mode='min',
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss' if not is_gan else 'g_loss',
            factor=0.5, patience=max(2, args.patience // 3), min_lr=1e-6, verbose=1, mode='min',
        ),
    ]

    # ModelCheckpoint only for standard models (GAN has custom loop)
    if not is_gan:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=best_weights_path, monitor='val_loss',
                save_best_only=True, save_weights_only=True, verbose=1,
            )
        )

    # ── Fit ───────────────────────────────────────────────────────────────────
    log.info(f"Starting training  epochs={args.epochs}  batch_size={args.batch_size}")
    t0 = time.time()

    if is_gan:
        # GAN needs a tf.data.Dataset (custom train_step)
        X_noisy_gan = X_noisy * 2.0 - 1.0
        X_clean_gan = X_clean * 2.0 - 1.0
        dataset = (
            tf.data.Dataset.from_tensor_slices((X_noisy_gan, X_clean_gan))
            .shuffle(2000).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)
        )
        history = model.fit(dataset, epochs=args.epochs, callbacks=callbacks)
    else:
        history = model.fit(
            X_noisy, X_clean,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
        )

    elapsed = time.time() - t0
    log.info(f"Training done in {elapsed:.1f}s")

    # ── Save weights ──────────────────────────────────────────────────────────
    final_path = os.path.join(weights_dir, 'final_weights.weights.h5')
    if is_gan:
        model.generator.save_weights(final_path)
    else:
        model.save_weights(final_path)
    log.info(f"Weights saved → {final_path}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    log.info("Computing evaluation metrics …")
    # Evaluate on the validation part (the last 10%, never used for training).
    # The first images were training images, which made the scores look better than they are.
    eval_noisy = X_noisy[-200:]
    eval_clean = X_clean[-200:]

    if is_gan:
        preds_raw = model.generator(eval_noisy * 2.0 - 1.0, training=False).numpy()
        preds     = (preds_raw + 1.0) / 2.0
    else:
        preds = model.predict(eval_noisy, verbose=0)

    metrics = compute_all(eval_clean, preds)
    log.info(f"  PSNR  : {metrics['psnr']:.2f} dB")
    log.info(f"  SSIM  : {metrics['ssim']:.4f}")
    log.info(f"  MAE   : {metrics['mae']:.4f}")

    with open(os.path.join(exp_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ── Visual outputs ────────────────────────────────────────────────────────
    log.info("Saving comparison images …")
    comp_preds = preds if not is_gan else preds
    save_path = _save_comparisons(eval_noisy, eval_clean, comp_preds, outputs_dir, n=4)
    log.info(f"  Saved → {save_path}")

    # Loss curve
    hist_dict = history.history
    _save_loss_curve(hist_dict, outputs_dir)
    log.info(f"  Loss curve → {outputs_dir}/loss_curve.png")

    log.info("=" * 60)
    log.info(f"All outputs in: {exp_dir}/")
    log.info("=" * 60)
