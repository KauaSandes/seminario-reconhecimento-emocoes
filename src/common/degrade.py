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
    raise NotImplementedError


def gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    """Desfoque gaussiano com núcleo ``ksize`` x ``ksize`` (ímpar: 3, 5 ou 7).

    Usa ``cv2.GaussianBlur(img, (ksize, ksize), 0)``. Com sigma 0, o OpenCV calcula
    ``sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`` (0,8, 1,1 e 1,4 para 3, 5 e 7). A
    borda é tratada por reflexão (``BORDER_REFLECT_101``).
    """
    raise NotImplementedError


def downsample(img: np.ndarray, size: int) -> np.ndarray:
    """Reduz a imagem para ``size`` x ``size`` e volta ao tamanho original.

    A redução usa ``INTER_AREA`` (média por área, que funciona como um filtro antialias
    antes da amostragem) e a ampliação usa ``INTER_LINEAR``. Assim a informação acima da
    nova frequência de Nyquist se perde, mas a entrada do modelo continua com 48x48.
    """
    raise NotImplementedError


def apply_degradation(X: np.ndarray, kind: Kind, level: float, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Aplica uma degradação a um lote uint8 [N, H, W] e devolve um novo array do mesmo formato.

    Para ``"noise"``, um único ``np.random.default_rng(seed)`` é usado no lote inteiro,
    na ordem das imagens. Para ``"none"``, devolve uma cópia sem alteração.

    Raises:
        ValueError: com ``kind`` desconhecido ou ``level`` inválido para o tipo.
    """
    raise NotImplementedError
