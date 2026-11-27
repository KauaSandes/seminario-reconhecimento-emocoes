"""Pré-processamento da Frente B (CLAUDE.md, seção 3).

Na FER-2013 o rosto já vem recortado em 48x48, então não há Viola-Jones aqui: cada imagem passa
por equalização de histograma (contraste) e por um filtro gaussiano leve (ruído de alta frequência).
"""
from __future__ import annotations

import numpy as np


def preprocess(X: np.ndarray, blur_ksize: int = 3) -> np.ndarray:
    """Equalização de histograma + gaussiano ``blur_ksize`` x ``blur_ksize`` em cada imagem uint8 [N, H, W].

    ``blur_ksize=0`` desliga o gaussiano. Devolve um novo array uint8 com a mesma forma.
    """
    import cv2

    X = np.asarray(X)
    if X.dtype != np.uint8 or X.ndim != 3:
        raise ValueError(f"esperado uint8 [N, H, W], recebido {X.dtype} {X.shape}")
    if blur_ksize and blur_ksize % 2 == 0:
        raise ValueError("blur_ksize deve ser ímpar (ou 0 para desligar)")
    out = np.empty_like(X)
    for i, img in enumerate(X):
        eq = cv2.equalizeHist(img)
        out[i] = cv2.GaussianBlur(eq, (blur_ksize, blur_ksize), 0) if blur_ksize else eq
    return out
