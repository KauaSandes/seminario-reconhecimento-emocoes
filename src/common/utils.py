"""Utilidades pequenas que as duas frentes compartilham."""
from __future__ import annotations

import functools
import hashlib
import os
import random
import sys
from pathlib import Path

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


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    """SHA-256 (hex) de um arquivo, lido em blocos."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@functools.lru_cache(maxsize=4)
def load_face_cascade(name: str = "haarcascade_frontalface_default.xml"):
    """Carrega (uma vez) um Haar cascade do OpenCV para o Viola-Jones."""
    import cv2

    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / name))
    if cascade.empty():
        raise RuntimeError(f"não consegui carregar o cascade {name} de {cv2.data.haarcascades}")
    return cascade


def detect_faces(gray: np.ndarray, *, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: tuple[int, int] = (30, 30)) -> np.ndarray:
    """Viola-Jones numa imagem uint8 em cinza. Devolve as caixas int [k, 4] (x, y, w, h)."""
    faces = load_face_cascade().detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                                 minSize=min_size)
    return np.asarray(faces, dtype=np.int64).reshape(-1, 4)


def ensure_utf8_stdio() -> None:
    """Força UTF-8 no stdout e no stderr quando a saída é redirecionada (os acentos não se perdem no Windows)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and not stream.isatty():
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
