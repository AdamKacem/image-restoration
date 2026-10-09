"""
Model registry — maps CLI --model name to build function.

Usage (in trainer):
    from src.models import get_model
    model = get_model('unet', img_size=128)
"""

from .cnn       import build_cnn
from .unet      import build_unet
from .unet_gan  import build_unet_gan
from .mwcnn     import build_mwcnn
from .vit       import build_vit

# ── Registry ─────────────────────────────────────────────────────────────────
# Maps CLI name → (build_fn, accepted_kwargs)
_REGISTRY = {
    'cnn':      build_cnn,
    'unet':     build_unet,
    'unet_gan': build_unet_gan,
    'mwcnn':    build_mwcnn,
    'vit':      build_vit,
}

AVAILABLE_MODELS = list(_REGISTRY.keys())


def get_model(name: str, **kwargs):
    """
    Instantiate a model by name, forwarding only the kwargs it accepts.

    Args:
        name   : one of AVAILABLE_MODELS.
        kwargs : img_size, dropout_rate, base_filters, etc.
                 Unknown kwargs are silently ignored per-model.
    """
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {AVAILABLE_MODELS}"
        )
    build_fn = _REGISTRY[name]

    import inspect
    sig     = inspect.signature(build_fn)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    # The CNN calls its width `filters`; let --base_filters control it too.
    if 'filters' in sig.parameters and 'base_filters' in kwargs:
        accepted['filters'] = kwargs['base_filters']
    return build_fn(**accepted)
