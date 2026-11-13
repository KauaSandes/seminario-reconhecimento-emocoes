"""Limpeza da FER-2013 e pesos de classe.

Nada é apagado. A limpeza gera um arquivo de decisão, ``splits/fer2013_clean.csv``
(versionado), com uma linha por imagem e as colunas ``id, split, label, keep, reason``,
e um relatório, ``reports/data_cleaning.md``.

Regras (só o treino perde imagens, e a validação e o teste nunca são alterados):

=========================  ==================================  ============================
Situação                   Treino                              Validação / teste
=========================  ==================================  ============================
Duplicata no mesmo split   fica a de menor id, as outras saem  ``flag:dup_within_split``
                           (``dup_within_split``)
Duplicata treino x val/te  sai do treino                       intacta
                           (``dup_cross_split_leak``)
Rótulos conflitantes       todas as cópias saem                ``flag:label_conflict``
                           (``label_conflict``)
Quase constante (std < t)  sai (``low_std``)                   ``flag:low_std``
Viola-Jones sem rosto      ``flag:no_face`` (não sai)          ``flag:no_face``
=========================  ==================================  ============================

Quando há mais de um motivo, eles vêm separados por ``;``. Os motivos com prefixo
``flag:`` só sinalizam e nunca removem.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: Motivos que removem a imagem do treino.
REMOVE_REASONS: tuple[str, ...] = ("dup_within_split", "dup_cross_split_leak", "label_conflict", "low_std")
#: Motivos que só sinalizam.
FLAG_REASONS: tuple[str, ...] = ("flag:dup_within_split", "flag:label_conflict", "flag:low_std", "flag:no_face")
DEFAULT_STD_THRESHOLD = 5.0
CLEAN_COLUMNS: tuple[str, ...] = ("id", "split", "label", "keep", "reason")


def pixel_hashes(X: np.ndarray) -> np.ndarray:
    """MD5 (hex) dos bytes de cada imagem uint8 [N, H, W]."""
    raise NotImplementedError


def detect_no_face(X: np.ndarray, upscale: int = 96, min_size: int = 24) -> np.ndarray:
    """Marca as imagens em que o Viola-Jones não encontra rosto (bool [N]).

    A imagem é ampliada para ``upscale`` x ``upscale`` antes da detecção. Em 48x48 o
    detector erra bastante, por isso o resultado só serve como sinalização.
    """
    raise NotImplementedError


def clean_fer2013(X: np.ndarray, y: np.ndarray, split: np.ndarray, ids: np.ndarray, *,
                  std_threshold: float = DEFAULT_STD_THRESHOLD, face_check: bool = True) -> pd.DataFrame:
    """Aplica as regras do módulo e devolve a tabela de decisão.

    Args:
        X, y, split, ids: saída de ``src.common.data.fer2013_arrays()`` (todas as imagens).
        std_threshold: imagens com desvio padrão dos pixels abaixo disto são "quase
            constantes".
        face_check: roda o Viola-Jones para sinalizar ``flag:no_face``.

    Returns:
        DataFrame com as colunas ``CLEAN_COLUMNS``, mais ``std`` e ``md5`` (que servem
        ao relatório e não vão para o CSV versionado), ordenado por ``id``.
    """
    raise NotImplementedError


def write_clean_csv(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Grava ``splits/fer2013_clean.csv`` (só as colunas ``CLEAN_COLUMNS``)."""
    raise NotImplementedError


def save_low_std_grid(X: np.ndarray, df: pd.DataFrame, path: Path | str | None = None,
                      std_threshold: float = DEFAULT_STD_THRESHOLD, max_images: int = 96) -> Path:
    """Grade das imagens quase constantes e das que estão perto do limiar, para conferência visual.

    O padrão é ``data/processed/inspection/fer2013_low_std_grid.png``, que fica fora do
    git porque contém imagens da base.
    """
    raise NotImplementedError


def write_report(df: pd.DataFrame, path: Path | str | None = None, *, std_threshold: float,
                 csv_sha256: str | None = None, face_check: bool = True,
                 grid_path: Path | str | None = None) -> Path:
    """Gera ``reports/data_cleaning.md`` com as contagens por motivo e classe e as distribuições."""
    raise NotImplementedError


def class_weights(y: np.ndarray, n_classes: int = 7) -> np.ndarray:
    """Pesos de classe pelo inverso da frequência, normalizados (esquema "balanced").

    ``w_c = N / (n_classes * n_c)``, igual ao ``class_weight="balanced"`` do
    scikit-learn. A média dos pesos ponderada pela frequência é 1. Serve para
    ``CrossEntropyLoss(weight=...)`` e para os classificadores da Frente B. A classe
    "nojo" (1) é a menor e recebe o maior peso. Uma classe ausente recebe peso 0.

    Returns:
        float64 [n_classes].
    """
    raise NotImplementedError
