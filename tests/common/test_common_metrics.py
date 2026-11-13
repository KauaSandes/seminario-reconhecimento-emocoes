"""Testes das métricas e do JSON de resultados (src/common/metrics.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.common import metrics

META = {"method": "majority", "track": "baseline", "dataset": "fer2013", "split": "val",
        "degradation": "none", "level": 0, "train_time_s": 0.0, "inference_ms_per_img": 0.01, "seed": 42}

# Exemplo de referência do contrato (CLAUDE.md, seção 5), com os valores trocados
CONTRACT_KEYS = ["method", "track", "dataset", "split", "degradation", "level", "accuracy", "f1_macro",
                 "f1_per_class", "confusion_matrix", "train_time_s", "inference_ms_per_img", "seed",
                 "git_commit"]


def test_record_has_exactly_contract_keys():
    rec = metrics.evaluate([0, 1, 2], [0, 1, 1], META)
    assert list(rec) == CONTRACT_KEYS == list(metrics.RESULT_KEYS)


def test_hand_computed_case():
    y_true = np.array([0, 0, 1, 1, 2, 3])
    y_pred = np.array([0, 1, 1, 1, 2, 0])
    rec = metrics.evaluate(y_true, y_pred, META)
    assert rec["accuracy"] == pytest.approx(4 / 6)
    # F1: classe 0 = 2*1/(2+2)=0.5; classe 1 = 2*2/(3+2)=0.8; classe 2 = 1; classe 3 = 0; 4..6 = 0
    assert rec["f1_per_class"] == pytest.approx([0.5, 0.8, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert rec["f1_macro"] == pytest.approx(2.3 / 7)
    cm = np.array(rec["confusion_matrix"])
    assert cm.shape == (7, 7) and cm.sum() == 6
    assert cm[0, 1] == 1 and cm[3, 0] == 1 and cm[1, 1] == 2  # linha = verdadeiro, coluna = previsto


def test_json_is_written_and_valid(tmp_path):
    rec = metrics.evaluate(np.arange(7), np.arange(7), {**META, "split": "test", "degradation": "noise", "level": 10})
    path = metrics.save_results(rec, tmp_path)
    assert path.name == "majority_fer2013_test_noise_10.json"
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert list(loaded) == CONTRACT_KEYS
    assert loaded["accuracy"] == 1.0 and loaded["level"] == 10 and loaded["seed"] == 42
    assert isinstance(loaded["git_commit"], str) and loaded["git_commit"]
    metrics.validate_record(loaded)


def test_git_commit_format():
    commit = metrics.git_commit()
    assert commit == "unknown" or len(commit.split("+")[0]) == 40


@pytest.mark.parametrize("change", [
    {"track": "deep"},                                  # track fora do contrato
    {"dataset": "jaffe"},
    {"split": "validation"},
    {"degradation": "noise", "level": 7},               # nível fora de LEVELS
    {"degradation": "blur", "level": 3},                # degradação fora do teste
    {"method": "hog svm"},                              # nome com espaço vira arquivo ruim
    {"train_time_s": -1},
])
def test_invalid_meta(change):
    with pytest.raises(ValueError):
        metrics.evaluate([0], [0], {**META, **change})


def test_missing_or_extra_meta_keys():
    meta = dict(META)
    del meta["seed"]
    with pytest.raises(ValueError, match="faltam"):
        metrics.evaluate([0], [0], meta)
    with pytest.raises(ValueError, match="sobram"):
        metrics.evaluate([0], [0], {**META, "fold": 3})


def test_invalid_labels():
    with pytest.raises(ValueError):
        metrics.evaluate([0, 7], [0, 1], META)
    with pytest.raises(ValueError):
        metrics.evaluate([0, 1], [0], META)


def test_save_rejects_tampered_record(tmp_path):
    rec = metrics.evaluate([0, 1], [0, 1], META)
    with pytest.raises(ValueError):
        metrics.save_results({**rec, "extra": 1}, tmp_path)
    with pytest.raises(ValueError):
        metrics.save_results({**rec, "f1_per_class": [0.0] * 6}, tmp_path)
