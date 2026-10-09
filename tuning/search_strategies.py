"""
search_strategies.py — parameter sampling for hyperparameter search.

Two strategies, zero external dependencies:
  - RandomSearch : sample N random configs from a parameter space.
  - GridSearch   : enumerate every combination (Cartesian product).

Parameter space format (same for both):
    {
        'lr':          {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-2},
        'batch_size':  {'type': 'choice',      'values': [8, 16, 32]},
        'dropout':     {'type': 'uniform',     'low': 0.1, 'high': 0.5},
        'optimizer':   {'type': 'choice',      'values': ['adam', 'adamw']},
        'num_layers':  {'type': 'int',         'low': 2,   'high': 8},
    }

Supported types:
  choice       — pick one element at random (also enumerable for grid)
  uniform      — continuous float in [low, high]
  log_uniform  — continuous float in [10^log10(low), 10^log10(high)]
  int          — random integer in [low, high]  (inclusive)

For GridSearch, only 'choice' values are enumerable.
'uniform' / 'log_uniform' / 'int' in grid mode are treated as a list of
discrete values defined via 'values' key (raise if absent).
"""

import random
import math
import itertools
from typing import List, Dict, Any


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_param(spec: dict, rng: random.Random) -> Any:
    t = spec['type']
    if t == 'choice':
        return rng.choice(spec['values'])
    if t == 'uniform':
        return rng.uniform(spec['low'], spec['high'])
    if t == 'log_uniform':
        log_low  = math.log10(spec['low'])
        log_high = math.log10(spec['high'])
        return 10 ** rng.uniform(log_low, log_high)
    if t == 'int':
        return rng.randint(int(spec['low']), int(spec['high']))
    raise ValueError(f"Unknown parameter type: '{t}'")


def _grid_values(name: str, spec: dict) -> List[Any]:
    """
    Return the discrete list of values to enumerate for this parameter.
    'choice' uses spec['values']; others require spec['values'] too.
    """
    if 'values' in spec:
        return list(spec['values'])
    raise ValueError(
        f"Parameter '{name}' of type '{spec['type']}' needs a 'values' list for grid search."
    )


# ── Strategy classes ──────────────────────────────────────────────────────────

class RandomSearch:
    """
    Sample `n_trials` configurations independently at random.
    Each call to .configs() returns a fresh list (reproducible with seed).
    """

    def __init__(self, space: Dict[str, dict], n_trials: int, seed: int = 42):
        self.space    = space
        self.n_trials = n_trials
        self.rng      = random.Random(seed)

    def configs(self) -> List[Dict[str, Any]]:
        results = []
        for _ in range(self.n_trials):
            cfg = {name: _sample_param(spec, self.rng)
                   for name, spec in self.space.items()}
            results.append(cfg)
        return results

    def __repr__(self):
        return f"RandomSearch(n_trials={self.n_trials}, params={list(self.space)})"


class GridSearch:
    """
    Enumerate every combination in the Cartesian product of all value lists.
    If the total number of combinations exceeds `max_trials`, a random
    sub-sample is returned (still deterministic with seed).
    """

    def __init__(self, space: Dict[str, dict], max_trials: int = 50, seed: int = 42):
        self.space      = space
        self.max_trials = max_trials
        self.rng        = random.Random(seed)

    def configs(self) -> List[Dict[str, Any]]:
        keys   = list(self.space)
        values = [_grid_values(k, self.space[k]) for k in keys]

        all_combos = [dict(zip(keys, combo))
                      for combo in itertools.product(*values)]

        if len(all_combos) <= self.max_trials:
            return all_combos

        # Sub-sample deterministically
        sampled = self.rng.sample(all_combos, self.max_trials)
        return sampled

    def __repr__(self):
        keys   = list(self.space)
        values = [_grid_values(k, self.space[k]) for k in keys]
        total  = 1
        for v in values:
            total *= len(v)
        return f"GridSearch(total_combinations={total}, max_trials={self.max_trials})"


# ── Default parameter spaces per model ───────────────────────────────────────
# These are used when --config is not provided.

DEFAULT_SPACES = {
    'cnn': {
        'lr':         {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
        'batch_size': {'type': 'choice',      'values': [8, 16, 32]},
        'optimizer':  {'type': 'choice',      'values': ['adam', 'adamw']},
        'dropout':    {'type': 'uniform',     'low': 0.0, 'high': 0.4},
        'loss':       {'type': 'choice',      'values': ['weighted_mse_ssim', 'mse']},
    },
    'unet': {
        'lr':          {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
        'batch_size':  {'type': 'choice',      'values': [8, 16, 32]},
        'optimizer':   {'type': 'choice',      'values': ['adam', 'adamw']},
        'dropout':     {'type': 'uniform',     'low': 0.1, 'high': 0.4},
        'base_filters':{'type': 'choice',      'values': [16, 32, 64]},
        'loss':        {'type': 'choice',      'values': ['weighted_mse_ssim', 'mse']},
        'ink_weight': {'type': 'choice',  'values': [2.0, 4.0, 6.0, 8.0]},
        'mse_alpha':  {'type': 'choice',  'values': [0.3, 0.5, 0.7]},
    },
    'unet_gan': {
        'lr':         {'type': 'log_uniform', 'low': 1e-5, 'high': 1e-3},
        'batch_size': {'type': 'choice',      'values': [8, 16]},
        'optimizer':  {'type': 'choice',      'values': ['adam']},
    },
    'mwcnn': {
        'lr':         {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-2},
        'batch_size': {'type': 'choice',      'values': [8, 16, 32]},
        'optimizer':  {'type': 'choice',      'values': ['adam', 'adamw']},
    },
    'vit': {
        'lr':         {'type': 'log_uniform', 'low': 1e-4, 'high': 1e-3},
        'batch_size': {'type': 'choice',      'values': [8, 16]},
        'optimizer':  {'type': 'choice',      'values': ['adam']},
        'embed_dim':  {'type': 'choice',      'values': [128, 256]},
        'num_layers': {'type': 'choice',      'values': [2, 4, 6]},
        'num_heads':  {'type': 'choice',      'values': [4, 8]},
        'mlp_dim':    {'type': 'choice',      'values': [256, 512]},
        'dropout':    {'type': 'uniform',     'low': 0.05, 'high': 0.3},
        'loss':       {'type': 'choice',      'values': ['weighted_mse_ssim', 'mse']},
        'ink_weight': {'type': 'choice',  'values': [2.0, 4.0, 6.0, 8.0]},
        'mse_alpha':  {'type': 'choice',  'values': [0.3, 0.5, 0.7]},
    },
}
