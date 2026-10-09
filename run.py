#!/usr/bin/env python
"""
run.py — single entry point for the image restoration pipeline.

Commands
────────
  generate   : create synthetic (noisy, clean) image pairs
  train      : train a model on generated data
  evaluate   : evaluate a saved experiment

Usage examples
──────────────
  python run.py generate --num_samples 500 --noise_level 0.3 --output data/synthetic/

  python run.py train \\
      --model unet \\
      --epochs 5 \\
      --batch_size 16 \\
      --lr 0.001 \\
      --dataset data/synthetic \\
      --exp_name exp_test

  python run.py evaluate --exp experiments/exp_test/ --n_eval 200

  python run.py tune --model unet --trials 10
  python run.py tune --config configs/tune_unet.yaml
"""

import argparse
import sys
import os

# ── make sure src/ is importable even if run from project root ─────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.models import AVAILABLE_MODELS


# ═════════════════════════════════════════════════════════════════════════════
# Argument parser
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Image Restoration Pipeline',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)

    # ── generate ─────────────────────────────────────────────────────────────
    gen = sub.add_parser('generate', help='Generate synthetic training data')
    gen.add_argument('--num_samples', type=int,   default=1000,
                     help='Number of image pairs to generate (default: 1000)')
    gen.add_argument('--img_size',   type=int,   default=128,
                     help='Image height = width in pixels (default: 128)')
    gen.add_argument('--noise_level', type=float, default=0.06,
                     help='Max Gaussian noise std (default: 0.06)')
    gen.add_argument('--corruption', type=str,   default='stamp',
                     choices=['stamp', 'blur', 'occlusion', 'all'],
                     help='Type of corruption to apply (default: stamp)')
    gen.add_argument('--output',     type=str,   default='data/synthetic/',
                     help='Output directory (default: data/synthetic/)')

    # ── train ─────────────────────────────────────────────────────────────────
    tr = sub.add_parser('train', help='Train a restoration model')

    # data & experiment
    tr.add_argument('--dataset',    type=str,   default='data/synthetic',
                    help='Path to dataset folder with X_noisy.npy / X_clean.npy')
    tr.add_argument('--exp_name',   type=str,   default='exp_001',
                    help='Experiment name → experiments/<exp_name>/')

    # model
    tr.add_argument('--model',      type=str,   default='unet',
                    choices=AVAILABLE_MODELS,
                    help=f'Model architecture (default: unet). Choices: {AVAILABLE_MODELS}')
    tr.add_argument('--img_size',   type=int,   default=128,
                    help='Input image size (default: 128)')

    # training hyper-parameters
    tr.add_argument('--epochs',     type=int,   default=5,
                    help='Number of training epochs (default: 5)')
    tr.add_argument('--batch_size', type=int,   default=16,
                    help='Batch size (default: 16)')
    tr.add_argument('--lr',         type=float, default=1e-4,
                    help='Learning rate (default: 1e-4)')
    tr.add_argument('--optimizer',  type=str,   default='adam',
                    choices=['adam', 'adamw', 'sgd', 'rmsprop'],
                    help='Optimizer (default: adam)')
    tr.add_argument('--loss',       type=str,   default='auto',
                    choices=['auto', 'weighted_mse_ssim', 'wavelet', 'mse', 'mae'],
                    help='"auto" picks the default loss per model (default: auto)')
    tr.add_argument('--patience',   type=int,   default=8,
                    help='EarlyStopping patience (default: 8)')

    # model-specific parameters
    tr.add_argument('--dropout',      type=float, default=0.3,
                    help='Dropout rate for UNet/CNN (default: 0.3)')
    tr.add_argument('--base_filters', type=int,   default=32,
                    help='Base filters for UNet/CNN (default: 32)')

    # ViT-specific
    tr.add_argument('--patch_size',   type=int,   default=8,
                    help='[ViT] Patch size (default: 8)')
    tr.add_argument('--embed_dim',    type=int,   default=256,
                    help='[ViT] Token embedding dimension (default: 256)')
    tr.add_argument('--num_heads',    type=int,   default=8,
                    help='[ViT] Attention heads (default: 8)')
    tr.add_argument('--mlp_dim',      type=int,   default=512,
                    help='[ViT] MLP hidden dimension (default: 512)')
    tr.add_argument('--num_layers',   type=int,   default=6,
                    help='[ViT] Number of transformer blocks (default: 6)')
    tr.add_argument('--ink_weight',   type=float, default=None,
                    help='[weighted_mse_ssim] extra weight on dark (ink) pixels (default: 4.0)')
    tr.add_argument('--mse_alpha',    type=float, default=None,
                    help='[weighted_mse_ssim] weight of the MSE term, SSIM gets 1 - mse_alpha (default: 0.5)')

    # ── evaluate ──────────────────────────────────────────────────────────────
    ev = sub.add_parser('evaluate', help='Evaluate a saved experiment')
    ev.add_argument('--exp',     type=str, required=True,
                    help='Path to experiment folder, e.g. experiments/exp_test/')
    ev.add_argument('--dataset', type=str, default=None,
                    help='Override dataset path (default: uses path from config.json)')
    ev.add_argument('--n_eval',  type=int, default=200,
                    help='Number of samples to evaluate (default: 200)')


    # ── tune ──────────────────────────────────────────────────────────────────
    tu = sub.add_parser('tune', help='Hyperparameter search (random or grid)')

    tu.add_argument('--model',       type=str,  default='unet',
                    choices=AVAILABLE_MODELS,
                    help='Model to tune (default: unet)')
    tu.add_argument('--trials',      type=int,  default=10,
                    help='Number of trials to run (default: 10)')
    tu.add_argument('--strategy',    type=str,  default='random',
                    choices=['random', 'grid'],
                    help='Search strategy: random | grid (default: random)')
    tu.add_argument('--config',      type=str,  default=None,
                    help='YAML config defining search space, e.g. configs/tune_unet.yaml')
    tu.add_argument('--tuning_name', type=str,  default=None,
                    help='Name for tuning run → experiments/tuning/<name>/')
    tu.add_argument('--tune_epochs', type=int,  default=3,
                    help='Epochs per trial — keep small for speed (default: 3)')
    tu.add_argument('--dataset',     type=str,  default=None,
                    help='Dataset path override (default: data/synthetic)')
    tu.add_argument('--img_size',    type=int,  default=None,
                    help='Image size override (default: 128)')
    tu.add_argument('--seed',        type=int,  default=42,
                    help='Random seed for reproducibility (default: 42)')

    return parser


# ═════════════════════════════════════════════════════════════════════════════
# Command handlers
# ═════════════════════════════════════════════════════════════════════════════

def cmd_generate(args) -> None:
    from src.data_generator import generate_data, save_data
    print(f"\n[generate] Generating {args.num_samples} samples "
          f"(corruption={args.corruption}, noise={args.noise_level}) …")
    X_noisy, X_clean = generate_data(
        num_samples=args.num_samples,
        img_size=args.img_size,
        noise_level=args.noise_level,
        corruption=args.corruption,
    )
    save_data(X_noisy, X_clean, args.output)
    print(f"[generate] Done. Data saved to {args.output}\n")


def cmd_train(args) -> None:
    # Resolve 'auto' loss before passing to trainer
    if args.loss == 'auto':
        args.loss = 'wavelet' if args.model == 'mwcnn' else 'weighted_mse_ssim'

    from src.trainer import train
    train(args)


def cmd_evaluate(args) -> None:
    from src.evaluator import evaluate
    evaluate(args)


def cmd_tune(args) -> None:
    from tuning.tuner import run_tuning
    run_tuning(args)


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser  = build_parser()
    args    = parser.parse_args()

    dispatch = {
        'generate': cmd_generate,
        'train':    cmd_train,
        'evaluate': cmd_evaluate,
        'tune':     cmd_tune,
    }
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
