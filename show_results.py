"""
show_results.py — pretty-print a tuning results.json to the terminal.

Usage:
    python show_results.py experiments/tuning/tune_unet_random/results.json
    python show_results.py experiments/tuning/tune_unet_random/results.json --sort psnr
    python show_results.py experiments/tuning/tune_unet_random/results.json --top 5
"""

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(description='Display tuning results')
    p.add_argument('results', help='Path to results.json')
    p.add_argument('--sort',  default='ssim',
                   choices=['ssim', 'psnr', 'mae'],
                   help='Sort trials by this metric (default: ssim)')
    p.add_argument('--top',   type=int, default=None,
                   help='Show only top N trials')
    args = p.parse_args()

    with open(args.results) as f:
        data = json.load(f)

    model    = data.get('model', '?')
    strategy = data.get('strategy', '?')
    metric   = data.get('metric', 'ssim')
    trials   = data.get('all_trials', [])

    print(f"\n{'═'*70}")
    print(f"  Tuning results — model={model}  strategy={strategy}")
    print(f"  Primary metric: {metric}   Total trials: {len(trials)}")
    print(f"{'═'*70}")

    # Filter successful
    ok  = [t for t in trials if t.get('success') and t.get('metrics')]
    bad = [t for t in trials if not t.get('success')]

    # Sort
    reverse = (args.sort != 'mae')  # lower is better for mae
    ok_sorted = sorted(ok, key=lambda t: t['metrics'].get(args.sort, -999),
                       reverse=reverse)

    if args.top:
        ok_sorted = ok_sorted[:args.top]

    # Header
    print(f"\n{'Rank':<5} {'Trial':<12} {'SSIM':>6} {'PSNR':>7} {'MAE':>7}  {'Key params'}")
    print('-' * 70)

    for rank, t in enumerate(ok_sorted, 1):
        m   = t['metrics']
        p_str = '  '.join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in t['params'].items())
        print(f"{rank:<5} {t['trial']:<12} "
              f"{m.get('ssim',0):>6.4f} "
              f"{m.get('psnr',0):>7.2f} "
              f"{m.get('mae',0):>7.4f}  "
              f"{p_str}")

    if bad:
        print(f"\n  Failed trials: {len(bad)}")
        for t in bad:
            print(f"    {t['trial']}  — {t.get('error','unknown error')[:80]}")

    # Best
    best = data.get('best')
    if best:
        print(f"\n{'─'*70}")
        print(f"  ★  Best: {best['trial']}  SSIM={best['metrics'].get('ssim',0):.4f}  "
              f"PSNR={best['metrics'].get('psnr',0):.2f}")
        print(f"     Params: {best['params']}")

    print(f"{'═'*70}\n")


if __name__ == '__main__':
    main()
