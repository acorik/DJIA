"""Gestión del dispositivo de cómputo (CUDA RTX 4070 si está disponible)."""
from __future__ import annotations
import functools

from . import config


@functools.lru_cache(maxsize=1)
def get_device() -> str:
    if config.FORCE_DEVICE in ("cuda", "cpu"):
        return config.FORCE_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@functools.lru_cache(maxsize=1)
def torch_module():
    import torch
    return torch
