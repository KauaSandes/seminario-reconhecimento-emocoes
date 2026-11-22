"""Teste ponta a ponta de ``python -m src.common.prepare_data`` com dados sintéticos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common import clean, data, prepare_data


@pytest.fixture
def sandbox(fer_csv, ckplus_fake, tmp_path, monkeypatch):
    """Redireciona todos os caminhos padrão para tmp_path (nada é escrito no repositório)."""
    monkeypatch.setattr(data, "FER2013_CSV", fer_csv.path)
    monkeypatch.setattr(data, "FER2013_CACHE", tmp_path / "processed" / "fer2013.npz")
    monkeypatch.setattr(data, "FER2013_CLEAN_CSV", tmp_path / "splits" / "fer2013_clean.csv")
    monkeypatch.setattr(data, "FER2013_SIZES", fer_csv.sizes)
    monkeypatch.setattr(clean, "FER2013_CLEAN_CSV", tmp_path / "splits" / "fer2013_clean.csv")
    monkeypatch.setattr(clean, "REPORT_PATH", tmp_path / "reports" / "data_cleaning.md")
    monkeypatch.setattr(clean, "INSPECTION_DIR", tmp_path / "processed" / "inspection")
    monkeypatch.setattr(data, "CKPLUS_RAW_DIR", ckplus_fake.root)
    monkeypatch.setattr(data, "CKPLUS_CACHE", tmp_path / "processed" / "ckplus.npz")
    monkeypatch.setattr(data, "CKPLUS_FOLDS_CSV", tmp_path / "splits" / "ckplus_folds.csv")
    monkeypatch.setattr(data, "detect_faces", lambda gray, **k: np.zeros((0, 4), dtype=np.int64))
    return tmp_path


def test_prepare_data_end_to_end(sandbox, fer_csv, capsys):
    assert prepare_data.main(["--skip-face-check"]) == 0
    out = capsys.readouterr().out
    assert "SHA-256" in out and "[CK+]" in out

    decision = pd.read_csv(sandbox / "splits" / "fer2013_clean.csv", keep_default_na=False)
    assert list(decision.columns) == ["id", "split", "label", "keep", "reason"]
    assert len(decision) == sum(fer_csv.sizes.values())
    assert decision.loc[decision["split"] != "train", "keep"].all()
    assert (sandbox / "reports" / "data_cleaning.md").is_file()
    assert (sandbox / "processed" / "inspection" / "fer2013_low_std_grid.png").is_file()
    assert (sandbox / "splits" / "ckplus_folds.csv").is_file()

    # os carregadores do contrato funcionam com os caminhos padrão depois do prepare_data
    X, y, ids = data.load_fer2013("train")
    assert X.shape[1:] == (48, 48) and len(X) < fer_csv.sizes["Training"]
    assert len(data.load_fer2013("test")[0]) == fer_csv.sizes["PrivateTest"]
    Xc, yc, _ = data.load_ckplus(0, "train", "image")
    assert Xc.dtype == np.uint8 and Xc.shape[1:] == (48, 48)


def test_prepare_data_fails_without_csv(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(data, "FER2013_CSV", sandbox / "nada.csv")
    assert prepare_data.main(["--skip-face-check"]) == 1
    assert "download_data" in capsys.readouterr().err


def test_prepare_data_rejects_wrong_counts(sandbox, monkeypatch, capsys):
    monkeypatch.setattr(data, "FER2013_SIZES", {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589})
    assert prepare_data.main(["--skip-face-check"]) == 1
    assert "oficiais" in capsys.readouterr().err
