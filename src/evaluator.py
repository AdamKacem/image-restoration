"""
Evaluator — loads a saved experiment folder and:
  1. Reads config.json to rebuild the same model.
  2. Loads best or final weights.
  3. Runs on the dataset specified in config (or overridden via CLI).
  4. Reports PSNR / SSIM / MAE.
  5. Saves fresh comparison images.
"""

import os
import json
import logging
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .models         import get_model
from .metrics        import compute_all
from .data_generator import load_data


def _load_config(exp_dir: str) -> dict:
    config_path = os.path.join(exp_dir, 'config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json found in {exp_dir}")
    with open(config_path) as f:
        return json.load(f)


def _get_logger(exp_dir: str):
    logger = logging.getLogger('evaluator')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s', datefmt='%H:%M:%S')
    ch  = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh  = logging.FileHandler(os.path.join(exp_dir, 'eval.log'))
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def evaluate(args) -> None:
    exp_dir = args.exp.rstrip('/')
    log     = _get_logger(exp_dir)
    config  = _load_config(exp_dir)
    log.info(f"Evaluating experiment: {exp_dir}")
    log.info(f"Original model: {config['model']}")

    # ── Rebuild model ─────────────────────────────────────────────────────────
    model_kwargs = dict(
        img_size     = config.get('img_size', 128),
        dropout_rate = config.get('dropout', 0.3),
        base_filters = config.get('base_filters', 32),
        patch_size   = config.get('patch_size', 8),
        embed_dim    = config.get('embed_dim', 256),
        num_heads    = config.get('num_heads', 8),
        mlp_dim      = config.get('mlp_dim', 512),
        num_layers   = config.get('num_layers', 6),
    )
    model_name = config['model']
    model      = get_model(model_name, **model_kwargs)
    is_gan     = (model_name == 'unet_gan')

    # ── Load weights ──────────────────────────────────────────────────────────
    weights_dir  = os.path.join(exp_dir, 'weights')
    best_path    = os.path.join(weights_dir, 'best_weights.weights.h5')
    final_path   = os.path.join(weights_dir, 'final_weights.weights.h5')
    weights_path = best_path if os.path.exists(best_path) else final_path

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"No weights found in {weights_dir}")

    if is_gan:
        model.generator.load_weights(weights_path)
    else:
        model.load_weights(weights_path)
    log.info(f"Loaded weights from {weights_path}")

    # ── Load data ─────────────────────────────────────────────────────────────
    data_dir = args.dataset if args.dataset else config.get('dataset', 'data/synthetic')
    log.info(f"Loading data from {data_dir} …")
    X_noisy, X_clean = load_data(data_dir)
    log.info(f"  {len(X_noisy)} samples loaded.")

    # ── Predict ───────────────────────────────────────────────────────────────
    n_eval = min(args.n_eval, len(X_noisy))
    eval_noisy = X_noisy[:n_eval]
    eval_clean = X_clean[:n_eval]

    if is_gan:
        preds_raw = model.generator(eval_noisy * 2.0 - 1.0, training=False).numpy()
        preds     = (preds_raw + 1.0) / 2.0
    else:
        preds = model.predict(eval_noisy, verbose=0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = compute_all(eval_clean, preds)
    log.info("─" * 40)
    log.info(f"  PSNR : {metrics['psnr']:.2f} dB")
    log.info(f"  SSIM : {metrics['ssim']:.4f}")
    log.info(f"  MAE  : {metrics['mae']:.4f}")
    log.info("─" * 40)

    eval_metrics_path = os.path.join(exp_dir, 'eval_metrics.json')
    with open(eval_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved → {eval_metrics_path}")

    # ── Visual comparison ─────────────────────────────────────────────────────
    out_dir = os.path.join(exp_dir, 'outputs', 'comparisons')
    os.makedirs(out_dir, exist_ok=True)
    n_show = min(4, n_eval)

    fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 10))
    labels  = ['Noisy (input)', 'Clean (target)', 'Prediction']
    sources = [eval_noisy, eval_clean, preds]

    for row, (label, src) in enumerate(zip(labels, sources)):
        for col in range(n_show):
            axes[row, col].imshow(src[col, :, :, 0], cmap='gray', vmin=0, vmax=1)
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=11)
            axes[row, col].axis('off')

    plt.suptitle(f'Evaluation — {model_name}  |  PSNR {metrics["psnr"]:.2f} dB  '
                 f'SSIM {metrics["ssim"]:.4f}', fontsize=12)
    plt.tight_layout()
    img_path = os.path.join(out_dir, 'eval_comparison.png')
    plt.savefig(img_path, dpi=100, bbox_inches='tight')
    plt.close()
    log.info(f"Comparison image saved → {img_path}")
