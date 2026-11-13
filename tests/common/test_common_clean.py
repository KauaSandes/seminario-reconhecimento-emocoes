"""Testes da limpeza da FER-2013 (src/common/clean.py) com problemas plantados."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.class_weight import compute_class_weight

from src.common import clean, data


@pytest.fixture
def arrays(fer_csv, tmp_path):
    return data.fer2013_arrays(fer_csv.path, tmp_path / "c.npz")


@pytest.fixture
def decision(arrays):
    return clean.clean_fer2013(arrays["X"], arrays["y"], arrays["split"], arrays["ids"], face_check=False)


def _reasons(df, i):
    return set(filter(None, df.loc[df["id"] == i, "reason"].item().split(";")))


def test_planted_problems_are_detected(decision, fer_csv):
    for reason, planted_ids in fer_csv.planted.items():
        for i in planted_ids:
            assert reason in _reasons(decision, i), (reason, i, _reasons(decision, i))


def test_removed_are_exactly_the_planted_train_rows(decision, fer_csv):
    planted = {i for r in clean.REMOVE_REASONS for i in fer_csv.planted.get(r, [])}
    assert set(decision.loc[~decision["keep"], "id"]) == planted
    assert decision.loc[decision["id"] == 0, "keep"].item()  # a primeira cópia fica


def test_val_and_test_are_never_changed(decision, fer_csv):
    ev = decision[decision["split"] != "train"]
    assert ev["keep"].all()
    assert len(ev) == fer_csv.sizes["PublicTest"] + fer_csv.sizes["PrivateTest"]
    assert not ev["reason"].str.split(";").explode().isin(clean.REMOVE_REASONS).any()


def test_no_leakage_after_cleaning(decision):
    kept_train = set(decision.loc[(decision["split"] == "train") & decision["keep"], "md5"])
    evals = set(decision.loc[decision["split"] != "train", "md5"])
    assert not kept_train & evals
    # e nenhuma duplicata sobra dentro do treino limpo
    kept = decision[(decision["split"] == "train") & decision["keep"]]
    assert not kept["md5"].duplicated().any()


def test_output_columns_and_csv(decision, tmp_path):
    path = clean.write_clean_csv(decision, tmp_path / "fer2013_clean.csv")
    df = pd.read_csv(path, keep_default_na=False)
    assert list(df.columns) == list(clean.CLEAN_COLUMNS)
    assert df["keep"].dtype == bool and len(df) == len(decision)
    assert np.array_equal(df["id"], np.sort(df["id"]))


def test_clean_csv_works_with_loader(fer_csv, decision, tmp_path):
    path = clean.write_clean_csv(decision, tmp_path / "fer2013_clean.csv")
    kw = dict(csv_path=fer_csv.path, cache_path=tmp_path / "c.npz", splits_csv=path)
    X, y, ids = data.load_fer2013("train", **kw)
    assert len(X) == fer_csv.sizes["Training"] - int((~decision["keep"]).sum())
    assert len(data.load_fer2013("val", **kw)[0]) == fer_csv.sizes["PublicTest"]
    assert len(data.load_fer2013("test", **kw)[0]) == fer_csv.sizes["PrivateTest"]


def test_threshold_controls_low_std(arrays):
    d0 = clean.clean_fer2013(arrays["X"], arrays["y"], arrays["split"], arrays["ids"],
                             std_threshold=0.0, face_check=False)
    assert not d0["reason"].str.contains("low_std").any()


def test_face_check_only_flags(arrays, monkeypatch):
    monkeypatch.setattr(clean, "detect_no_face", lambda X: np.ones(len(X), dtype=bool))
    d = clean.clean_fer2013(arrays["X"], arrays["y"], arrays["split"], arrays["ids"], face_check=True)
    assert d["reason"].str.contains("flag:no_face").all()
    base = clean.clean_fer2013(arrays["X"], arrays["y"], arrays["split"], arrays["ids"], face_check=False)
    assert np.array_equal(d["keep"], base["keep"])


def test_detect_no_face_runs_on_noise(arrays):
    out = clean.detect_no_face(arrays["X"][:5])
    assert out.dtype == bool and out.shape == (5,)


def test_report_and_grid(arrays, decision, tmp_path):
    grid = clean.save_low_std_grid(arrays["X"], decision, tmp_path / "inspection" / "fer2013_low_std_grid.png")
    assert grid.is_file() and (grid.parent / "fer2013_low_std_hist.png").is_file()
    report = clean.write_report(decision, tmp_path / "data_cleaning.md", std_threshold=5.0,
                                csv_sha256="abc", face_check=False)
    text = report.read_text(encoding="utf-8")
    for needle in ("Resumo por conjunto", "dup_cross_split_leak", "Distribuição de classes", "abc", "nojo"):
        assert needle in text


def test_class_weights_matches_sklearn():
    y = np.array([0] * 50 + [1] * 5 + [2] * 20 + [3] * 90 + [4] * 30 + [5] * 25 + [6] * 40)
    expected = compute_class_weight("balanced", classes=np.arange(7), y=y)
    w = clean.class_weights(y)
    assert w.shape == (7,) and np.allclose(w, expected)
    assert w.argmax() == 1  # nojo é a menor classe
    assert np.isclose((w[y]).mean(), 1.0)


def test_class_weights_missing_class_warns():
    with pytest.warns(UserWarning):
        w = clean.class_weights(np.array([0, 0, 1, 2, 3, 4, 5]))
    assert w[6] == 0
    with pytest.raises(ValueError):
        clean.class_weights(np.array([0, 7]))
