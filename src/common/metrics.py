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

No CK+, junte as previsões dos 10 folds (cada amostra é prevista uma vez, no fold em que
está no teste) e chame ``evaluate`` uma vez só, com ``dataset="ckplus"`` e ``split="test"``.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from src.common.degrade import validate_level

NUM_CLASSES = 7
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
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
_METHOD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def git_commit() -> str:
    """Hash do commit atual (``git rev-parse HEAD``).

    Leva o sufixo ``+dirty`` se houver alterações não commitadas em arquivos
    versionados, e vale ``"unknown"`` quando não há git.
    """
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                               capture_output=True, text=True, check=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{head}+dirty" if dirty else head


def _as_int_level(level) -> int:
    if isinstance(level, bool) or not isinstance(level, (int, float, np.integer, np.floating)):
        raise ValueError(f"level deve ser numérico, não {level!r}")
    if float(level) != int(level):
        raise ValueError(f"level deve ser inteiro, não {level!r}")
    return int(level)


def _check_meta(meta: dict) -> dict:
    missing = [k for k in META_KEYS if k not in meta]
    extra = [k for k in meta if k not in META_KEYS]
    if missing or extra:
        raise ValueError(f"meta inválido: faltam {missing}, sobram {extra}. Esperado: {list(META_KEYS)}")
    out = dict(meta)
    if not isinstance(out["method"], str) or not _METHOD_RE.match(out["method"]):
        raise ValueError(f"method deve ser um nome simples (letras, números, _ . -), não {out['method']!r}")
    for key, allowed in (("track", TRACKS), ("dataset", DATASETS), ("split", EVAL_SPLITS)):
        if out[key] not in allowed:
            raise ValueError(f"{key} deve ser um de {allowed}, não {out[key]!r}")
    out["level"] = _as_int_level(out["level"])
    validate_level(out["degradation"], out["level"])
    if out["degradation"] != "none" and out["split"] != "test":
        raise ValueError("degradações são aplicadas só no teste (contrato, seção 5)")
    for key in ("train_time_s", "inference_ms_per_img"):
        value = float(out[key])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} deve ser um número >= 0, não {out[key]!r}")
        out[key] = value
    out["seed"] = int(out["seed"])
    return out


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, meta: dict) -> dict:
    """Calcula as métricas e monta o registro completo do JSON.

    Args:
        y_true, y_pred: rótulos inteiros 0..6.
        meta: as chaves de ``META_KEYS``. ``degradation`` e ``level`` usam os valores
            de ``src.common.degrade.LEVELS`` (sem degradação: ``"none"`` e ``0``).

    Returns:
        Dicionário com exatamente as chaves de ``RESULT_KEYS``: acurácia, F1 macro, F1
        por classe (7 valores), matriz de confusão 7x7 (linhas = verdadeiro, colunas =
        previsto) e ``git_commit``. O F1 macro é a média dos 7 F1 por classe, e uma classe
        sem amostras nem previsões conta como 0.

    Raises:
        ValueError: com chave faltando ou valor fora do contrato.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    meta = _check_meta(meta)
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError(f"y_true e y_pred devem ter o mesmo tamanho (> 0): {y_true.shape} x {y_pred.shape}")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if arr.min() < 0 or arr.max() >= NUM_CLASSES:
            raise ValueError(f"{name} com rótulos fora de 0..{NUM_CLASSES - 1}")

    labels = list(range(NUM_CLASSES))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    record = {
        **{k: meta[k] for k in ("method", "track", "dataset", "split", "degradation", "level")},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(np.mean(f1)),
        "f1_per_class": [float(v) for v in f1],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
        "train_time_s": meta["train_time_s"],
        "inference_ms_per_img": meta["inference_ms_per_img"],
        "seed": meta["seed"],
        "git_commit": git_commit(),
    }
    validate_record(record)
    return record


def validate_record(record: dict) -> None:
    """Confere se ``record`` segue exatamente o formato do contrato (ou dá ``ValueError``)."""
    if set(record) != set(RESULT_KEYS):
        raise ValueError(f"chaves diferentes do contrato: faltam {sorted(set(RESULT_KEYS) - set(record))}, "
                         f"sobram {sorted(set(record) - set(RESULT_KEYS))}")
    _check_meta({k: record[k] for k in META_KEYS})
    for key in ("accuracy", "f1_macro"):
        if not isinstance(record[key], float) or not 0.0 <= record[key] <= 1.0:
            raise ValueError(f"{key} deve ser float em [0, 1], não {record[key]!r}")
    f1 = record["f1_per_class"]
    if not isinstance(f1, list) or len(f1) != NUM_CLASSES or not all(0.0 <= float(v) <= 1.0 for v in f1):
        raise ValueError("f1_per_class deve ser uma lista com 7 valores em [0, 1]")
    cm = record["confusion_matrix"]
    if (not isinstance(cm, list) or len(cm) != NUM_CLASSES
            or any(not isinstance(r, list) or len(r) != NUM_CLASSES for r in cm)
            or any(not isinstance(v, int) or v < 0 for r in cm for v in r)):
        raise ValueError("confusion_matrix deve ser uma lista 7x7 de inteiros >= 0")
    if not isinstance(record["git_commit"], str) or not record["git_commit"]:
        raise ValueError("git_commit deve ser uma string não vazia")


def results_filename(record: dict) -> str:
    return f"{record['method']}_{record['dataset']}_{record['split']}_{record['degradation']}_{record['level']}.json"


def save_results(record: dict, out_dir: Path | str | None = None) -> Path:
    """Valida e grava ``record`` em ``results/{method}_{dataset}_{split}_{degradation}_{level}.json``.

    Returns:
        Caminho do arquivo gravado (sobrescreve se já existir).
    """
    validate_record(record)
    out = Path(out_dir) if out_dir is not None else RESULTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / results_filename(record)
    ordered = {k: record[k] for k in RESULT_KEYS}
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
