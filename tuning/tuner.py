"""
tuner.py — hyperparameter search orchestrator.

Design principle:
  The tuner does NOT duplicate training logic.
  It builds a standard argparse.Namespace (same shape as `python run.py train …`)
  and calls src.trainer.train() directly for each trial.

  This means every future change to the trainer is automatically reflected
  in tuning runs.

Folder layout produced:
  experiments/tuning/<tuning_name>/
      trial_000/   ← normal experiment folder (config.json, weights/, outputs/, …)
      trial_001/
      …
      results.json ← all trials + scores + best config
      tuning.log   ← timestamped log of the whole search
"""

import os
import json
import copy
import time
import logging
import argparse
from typing import Dict, Any, List

from .search_strategies import RandomSearch, GridSearch, DEFAULT_SPACES


# ── Logger ────────────────────────────────────────────────────────────────────

def _setup_logger(log_path: str, name: str = 'tuner') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(levelname)s  %(message)s',
                            datefmt='%H:%M:%S')
    for h in [logging.StreamHandler(),
              logging.FileHandler(log_path)]:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


# ── Default base args ─────────────────────────────────────────────────────────

_BASE_TRAIN_ARGS = dict(
    # data
    dataset    = 'data/synthetic',
    img_size   = 128,
    # training
    epochs     = 3,          # low default for speed; overridden by --tune_epochs
    batch_size = 16,
    lr         = 1e-4,
    optimizer  = 'adam',
    loss       = 'weighted_mse_ssim',
    patience   = 5,
    # model-specific
    dropout      = 0.3,
    base_filters = 32,
    patch_size   = 8,
    embed_dim    = 256,
    num_heads    = 8,
    mlp_dim      = 512,
    num_layers   = 6,

    ink_weight = 4.0,
    mse_alpha  = 0.5,
    ssim_alpha = 0.5,
)


def _make_train_args(model: str, exp_name: str, overrides: Dict[str, Any]) -> argparse.Namespace:
    """
    Build a Namespace identical to what `run.py train` would produce,
    applying `overrides` on top of the base defaults.
    """
    args_dict = copy.deepcopy(_BASE_TRAIN_ARGS)
    args_dict['model']    = model
    args_dict['exp_name'] = exp_name
    args_dict.update(overrides)

    # Resolve 'auto' loss
    if args_dict.get('loss') == 'auto':
        args_dict['loss'] = 'wavelet' if model == 'mwcnn' else 'weighted_mse_ssim'

    return argparse.Namespace(**args_dict)


# ── Score extraction ──────────────────────────────────────────────────────────

def _read_metrics(exp_dir: str) -> Dict[str, float]:
    """Read metrics.json from a completed trial. Returns empty dict on failure."""
    path = os.path.join(exp_dir, 'metrics.json')
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Main tuner ────────────────────────────────────────────────────────────────

def run_tuning(args) -> None:
    """
    Entry point called by run.py tune.
    `args` is the parsed Namespace from the tune sub-parser.
    """
    # ── Load / build parameter space ─────────────────────────────────────────
    if args.config:
        space, base_overrides = _load_yaml_config(args.config, args.model)
    else:
        space         = DEFAULT_SPACES.get(args.model, {})
        base_overrides = {}

    # CLI overrides (always win)
    if args.dataset:
        base_overrides['dataset'] = args.dataset
    if args.tune_epochs:
        base_overrides['epochs'] = args.tune_epochs
    if args.img_size:
        base_overrides['img_size'] = args.img_size

    # ── Build strategy ────────────────────────────────────────────────────────
    strategy_name = args.strategy.lower()
    if strategy_name == 'random':
        strategy = RandomSearch(space, n_trials=args.trials, seed=args.seed)
    elif strategy_name == 'grid':
        strategy = GridSearch(space, max_trials=args.trials, seed=args.seed)
    else:
        raise ValueError(f"Unknown strategy '{args.strategy}'. Choose: random, grid.")

    configs: List[Dict[str, Any]] = strategy.configs()
    n_trials = len(configs)

    # ── Setup tuning experiment folder ────────────────────────────────────────
    tuning_name = args.tuning_name or f"tune_{args.model}_{strategy_name}"
    tuning_dir  = os.path.join('experiments', 'tuning', tuning_name)
    os.makedirs(tuning_dir, exist_ok=True)

    log = _setup_logger(os.path.join(tuning_dir, 'tuning.log'))
    log.info("=" * 60)
    log.info(f"Tuning  model={args.model}  strategy={strategy_name}  trials={n_trials}")
    log.info(f"Output dir: {tuning_dir}")
    log.info(f"Strategy: {strategy}")
    log.info("=" * 60)

    # ── Run trials ────────────────────────────────────────────────────────────
    all_results = []

    for i, trial_cfg in enumerate(configs):
        trial_name = f"trial_{i:03d}"
        exp_name   = f"tuning/{tuning_name}/{trial_name}"
        exp_dir    = os.path.join('experiments', exp_name)

        log.info(f"\n{'─'*50}")
        log.info(f"Trial {i+1}/{n_trials}  →  {trial_name}")
        log.info(f"  Params: {trial_cfg}")

        # Merge base overrides then trial-specific params
        overrides = {**base_overrides, **trial_cfg}

        train_args = _make_train_args(
            model    = args.model,
            exp_name = exp_name,
            overrides= overrides,
        )

        t0      = time.time()
        success = True
        error   = None

        try:
            # ── THE ONLY CALL TO THE EXISTING TRAINER ──
            from src.trainer import train as _train
            _train(train_args)
        except Exception as e:
            success = False
            error   = str(e)
            log.error(f"  Trial {trial_name} FAILED: {e}")

        elapsed = time.time() - t0
        metrics = _read_metrics(exp_dir) if success else {}

        score = metrics.get('ssim', None)   # primary metric: SSIM (higher = better)

        result_entry = {
            'trial':   trial_name,
            'params':  trial_cfg,
            'metrics': metrics,
            'score':   score,
            'elapsed': round(elapsed, 1),
            'success': success,
        }
        if error:
            result_entry['error'] = error

        all_results.append(result_entry)

        if success and metrics:
            log.info(f"  ✓  PSNR={metrics.get('psnr', 0):.2f}  "
                     f"SSIM={metrics.get('ssim', 0):.4f}  "
                     f"MAE={metrics.get('mae', 0):.4f}  "
                     f"({elapsed:.1f}s)")

    # ── Find best ─────────────────────────────────────────────────────────────
    successful = [r for r in all_results if r['success'] and r['score'] is not None]

    best = None
    if successful:
        best = max(successful, key=lambda r: r['score'])
        log.info("\n" + "=" * 60)
        log.info("TUNING COMPLETE")
        log.info(f"  Best trial : {best['trial']}")
        log.info(f"  Best SSIM  : {best['score']:.4f}")
        log.info(f"  Best params: {best['params']}")
        log.info("=" * 60)
    else:
        log.warning("No successful trials to report.")

    # ── Save results summary ──────────────────────────────────────────────────
    summary = {
        'model':     args.model,
        'strategy':  strategy_name,
        'n_trials':  n_trials,
        'metric':    'ssim',
        'best':      best,
        'all_trials':all_results,
    }
    results_path = os.path.join(tuning_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info(f"Results saved → {results_path}")

    # ── Save best params as a ready-to-use train command ─────────────────────
    if best:
        _write_best_command(args.model, best['params'], base_overrides, tuning_dir, log)


# ── YAML config loader ────────────────────────────────────────────────────────

def _load_yaml_config(config_path: str, model_fallback: str):
    """
    Load a YAML tuning config.  Returns (space, base_overrides).
    Requires PyYAML (pip install pyyaml).

    Expected YAML structure:
        model: unet
        base:
          dataset: data/synthetic
          epochs: 3
        space:
          lr:
            type: log_uniform
            low: 0.0001
            high: 0.01
          batch_size:
            type: choice
            values: [8, 16, 32]
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for --config. Install with: pip install pyyaml"
        )

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    space          = cfg.get('space', {})
    base_overrides = cfg.get('base', {})

    return space, base_overrides


# ── Helper: write a ready-to-run best command ─────────────────────────────────

def _write_best_command(model: str, best_params: dict,
                        base_overrides: dict, tuning_dir: str,
                        log: logging.Logger) -> None:
    parts = [f"python run.py train --model {model}"]

    combined = {**base_overrides, **best_params}
    for k, v in combined.items():
        if k in ('model',):
            continue
        parts.append(f"  --{k} {v}")

    cmd = " \\\n".join(parts)
    cmd_path = os.path.join(tuning_dir, 'best_command.sh')
    with open(cmd_path, 'w') as f:
        f.write("#!/bin/bash\n# Best configuration found by tuning\n")
        f.write(cmd + "\n")
    log.info(f"Best train command → {cmd_path}")
    log.info(f"\nRun best config:\n{cmd}")
