"""Descritores da Frente B. Por enquanto: HOG e pixels brutos (linha de base do contrato).

Gabor, LBP e DCT entram nos próximos passos, com a mesma interface: ``f(X uint8 [N, H, W]) -> float32 [N, D]``.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

#: Parâmetros do HOG (Dalal e Triggs, 2005). Em 48x48: 6x6 células, 5x5 blocos x 2x2 células x 9 = 900 dimensões.
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")


def hog_features(X: np.ndarray) -> np.ndarray:
    """Histogramas de gradientes orientados de cada imagem."""
    from skimage.feature import hog

    return np.stack([hog(img, **HOG_PARAMS, feature_vector=True) for img in X]).astype(np.float32)


def pixel_features(X: np.ndarray) -> np.ndarray:
    """Os pixels em [0, 1], achatados (linha de base "pixels brutos + SVM")."""
    return (np.asarray(X).reshape(len(X), -1) / 255.0).astype(np.float32)


FEATURES: dict[str, Callable[[np.ndarray], np.ndarray]] = {"hog": hog_features, "pixels": pixel_features}
