"""Degradações para os testes de robustez (contrato, CLAUDE.md seção 5).

As degradações são aplicadas **só no teste**, e as duas frentes usam exatamente estas
funções. Para um conjunto inteiro, use ``apply_degradation``: ela usa um único gerador
com a semente fixa, então o ruído muda de imagem para imagem e sai idêntico nas duas
frentes.

================  ======================  =============  ==================
``kind``          função                  ``level``      valores do projeto
================  ======================  =============  ==================
``"none"``        (identidade)            0              0
``"noise"``       ``add_gaussian_noise``  sigma          5, 10, 20
``"blur"``        ``gaussian_blur``       ksize          3, 5, 7
``"resolution"``  ``downsample``          lado reduzido  24, 12
================  ======================  =============  ==================

Os valores de ``kind`` e ``level`` são os mesmos dos campos ``degradation`` e ``level``
do JSON de resultados.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

Kind = Literal["none", "noise", "blur", "resolution"]
KINDS: tuple[str, ...] = ("none", "noise", "blur", "resolution")
#: Níveis usados no projeto para cada degradação.
LEVELS: dict[str, tuple[int, ...]] = {"none": (0,), "noise": (5, 10, 20), "blur": (3, 5, 7), "resolution": (24, 12)}
DEFAULT_SEED = 42


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator | None = None,
                       seed: int = DEFAULT_SEED) -> np.ndarray:
    """Soma ruído gaussiano de média 0 e desvio ``sigma`` e recorta para 0..255.

    Args:
        img: uint8 2D (ou qualquer forma). A saída tem a mesma forma e é uint8.
        sigma: desvio padrão em níveis de cinza (5, 10 ou 20 no projeto).
        rng: gerador a usar. Se ``None``, cria ``np.random.default_rng(seed)``, de modo
            que a mesma imagem e a mesma semente sempre dão o mesmo resultado.
    """
    if sigma < 0:
        raise ValueError(f"sigma deve ser >= 0, não {sigma}")
    img = _as_uint8(img)
    rng = np.random.default_rng(seed) if rng is None else rng
    noise = rng.normal(0.0, float(sigma), size=img.shape).astype(np.float32)
    return np.clip(np.rint(img.astype(np.float32) + noise), 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Desfoque gaussiano com núcleo ``ksize`` x ``ksize`` (ímpar: 3, 5 ou 7).

    Usa ``cv2.GaussianBlur(img, (ksize, ksize), 0)``. Com sigma 0, o OpenCV calcula
    ``sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`` (0,8, 1,1 e 1,4 para 3, 5 e 7). A
    borda é tratada por reflexão (``BORDER_REFLECT_101``).
    """
    import cv2

    ksize = int(ksize)
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize deve ser ímpar e positivo, não {ksize}")
    img = _as_uint8(img, ndim=2)
    return cv2.GaussianBlur(img, (ksize, ksize), 0, borderType=cv2.BORDER_REFLECT_101)


def downsample(img: np.ndarray, size: int) -> np.ndarray:
    """Reduz a imagem para ``size`` x ``size`` e volta ao tamanho original.

    A redução usa ``INTER_AREA`` (média por área, que funciona como um filtro antialias
    antes da amostragem) e a ampliação usa ``INTER_LINEAR``. Assim a informação acima da
    nova frequência de Nyquist se perde, mas a entrada do modelo continua com 48x48.
    """
    import cv2

    img = _as_uint8(img, ndim=2)
    h, w = img.shape
    size = int(size)
    if not 1 <= size < min(h, w):
        raise ValueError(f"size deve estar entre 1 e {min(h, w) - 1} (menor que a imagem {h}x{w}), não {size}")
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _as_uint8(img: np.ndarray, ndim: int | None = None) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype != np.uint8:
        raise TypeError(f"esperado uint8, recebido {img.dtype}")
    if ndim is not None and img.ndim != ndim:
        raise ValueError(f"esperada imagem {ndim}D, recebida forma {img.shape}")
    return img


def validate_level(kind: str, level: float) -> None:
    """Confere se ``(kind, level)`` é uma combinação do projeto (ver ``LEVELS``)."""
    if kind not in LEVELS:
        raise ValueError(f"degradação desconhecida {kind!r}; use uma de {KINDS}")
    if level not in LEVELS[kind]:
        raise ValueError(f"nível {level!r} inválido para {kind!r}; use um de {LEVELS[kind]}")


def apply_degradation(X: np.ndarray, kind: Kind, level: float, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Aplica uma degradação a um lote uint8 [N, H, W] e devolve um novo array do mesmo formato.

    Para ``"noise"``, um único ``np.random.default_rng(seed)`` é usado no lote inteiro,
    na ordem das imagens. Para ``"none"``, devolve uma cópia sem alteração.

    Raises:
        ValueError: com ``kind`` desconhecido ou ``level`` fora de ``LEVELS[kind]``.
    """
    validate_level(kind, level)
    X = _as_uint8(X, ndim=3)
    if kind == "none":
        return X.copy()
    if kind == "noise":
        rng = np.random.default_rng(seed)
        return np.stack([add_gaussian_noise(img, level, rng=rng) for img in X]) if len(X) else X.copy()
    fn = gaussian_blur if kind == "blur" else downsample
    return np.stack([fn(img, level) for img in X]) if len(X) else X.copy()
