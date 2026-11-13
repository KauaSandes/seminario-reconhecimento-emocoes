"""Métricas e gravação dos resultados (contrato, CLAUDE.md seção 5).

Toda execução avaliada termina gravando um JSON em ``results/`` neste formato::

    {"method": "resnet18", "track": "cnn", "dataset": "fer2013", "split": "test",
     "degradation": "noise", "level": 10, "accuracy": 0.0, "f1_macro": 0.0,
     "f1_per_class": [0,0,0,0,0,0,0], "confusion_matrix": [[...]],
     "train_time_s": 0.0, "inference_ms_per_img": 0.0, "seed": 42, "git_commit": "..."}

Uso::

    from src.common.metrics import evaluate, save_results

    meta = {"method": "hog_linearsvc", "track": "classic", "dataset": "fer2013",
            "split": "val", "degradation": "none", "level": 0,
            "train_time_s": 12.3, "inference_ms_per_img": 0.4, "seed": 42}
    path = save_results(evaluate(y_true, y_pred, meta))
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: Chaves do JSON, na ordem do contrato.
RESULT_KEYS: tuple[str, ...] = (
    "method", "track", "dataset", "split", "degradation", "level",
    "accuracy", "f1_macro", "f1_per_class", "confusion_matrix",
    "train_time_s", "inference_ms_per_img", "seed", "git_commit",
)
#: Chaves que quem chama precisa passar em ``meta``.
META_KEYS: tuple[str, ...] = (
    "method", "track", "dataset", "split", "degradation", "level",
    "train_time_s", "inference_ms_per_img", "seed",
)
TRACKS: tuple[str, ...] = ("cnn", "classic", "baseline")
DATASETS: tuple[str, ...] = ("fer2013", "ckplus")
EVAL_SPLITS: tuple[str, ...] = ("train", "val", "test")
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def git_commit() -> str:
    """Hash do commit atual (``git rev-parse HEAD``).

    Leva o sufixo ``+dirty`` se houver alterações não commitadas em arquivos
    versionados, e vale ``"unknown"`` quando não há git.
    """
    raise NotImplementedError


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, meta: dict) -> dict:
    """Calcula as métricas e monta o registro completo do JSON.

    Args:
        y_true, y_pred: rótulos inteiros 0..6.
        meta: as chaves de ``META_KEYS``. ``degradation`` e ``level`` usam os valores
            de ``src.common.degrade.LEVELS`` (sem degradação: ``"none"`` e ``0``).

    Returns:
        Dicionário com exatamente as chaves de ``RESULT_KEYS``: acurácia, F1 macro, F1
        por classe (7 valores), matriz de confusão 7x7 (linhas = verdadeiro, colunas =
        previsto) e ``git_commit``.

    Raises:
        ValueError: com chave faltando ou valor fora do contrato.
    """
    raise NotImplementedError


def validate_record(record: dict) -> None:
    """Confere se ``record`` segue exatamente o formato do contrato (ou dá ``ValueError``)."""
    raise NotImplementedError


def save_results(record: dict, out_dir: Path | str | None = None) -> Path:
    """Valida e grava ``record`` em ``results/{method}_{dataset}_{split}_{degradation}_{level}.json``.

    Returns:
        Caminho do arquivo gravado (sobrescreve se já existir).
    """
    raise NotImplementedError
