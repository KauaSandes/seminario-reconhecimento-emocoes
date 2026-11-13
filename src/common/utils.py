"""Utilidades pequenas que as duas frentes compartilham."""
from __future__ import annotations

import os
import random
import sys

import numpy as np

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Fixa a semente de ``random``, ``numpy`` e, se instalado, do ``torch`` (CLAUDE.md, seção 5).

    Com ``deterministic=True``, também liga ``cudnn.deterministic`` e desliga o
    ``cudnn.benchmark``.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_utf8_stdio() -> None:
    """Força UTF-8 no stdout e no stderr quando a saída é redirecionada (os acentos não se perdem no Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
