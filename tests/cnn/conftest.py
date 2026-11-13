"""Dados falsos para os testes da Frente A (``src/cnn``).

Os módulos de ``src.common`` são os reais. Só a ORIGEM dos dados é trocada, para os
testes não dependerem dos arquivos baixados:

* ``fake_data``: ``load_fer2013`` devolve arrays aleatórios no formato do contrato
  (X uint8 [N, 48, 48], y int64 0..6, ids int64) e ``load_ckplus`` usa o carregador real
  sobre a pasta falsa do CK+ da Frente B (``tests/common/conftest.py``).
"""
from __future__ import annotations

import functools

import numpy as np
import pytest

from src.common import data
from tests.common.conftest import make_ckplus_fake

FAKE_SIZES = {"train": 140, "val": 35, "test": 35}


def fake_fer2013(split: str, clean: bool = True, **_):
    rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
    n = FAKE_SIZES[split]
    X = rng.integers(0, 256, size=(n, 48, 48), dtype=np.uint8)
    y = np.resize(np.arange(7), n)[rng.permutation(n)].astype(np.int64)  # todas as classes
    offset = {"train": 0, "val": 1000, "test": 2000}[split]
    return X, y, np.arange(offset, offset + n, dtype=np.int64)


@pytest.fixture(scope="session")
def ckplus_kw(tmp_path_factory):
    """Cache e folds do CK+ falso, montados uma vez por sessão com o código da Frente B."""
    root = tmp_path_factory.mktemp("ckplus")
    fake = make_ckplus_fake(root / "raw")
    mp = pytest.MonkeyPatch()
    mp.setattr(data, "detect_faces", lambda gray, **k: np.zeros((0, 4), dtype=np.int64))
    cache = data.build_ckplus_cache(fake.root, root / "ckplus.npz")
    folds = data.make_ckplus_folds(cache, root / "ckplus_folds.csv")
    mp.undo()
    return dict(raw_dir=fake.root, cache_path=cache, folds_csv=folds)


@pytest.fixture
def fake_data(monkeypatch, ckplus_kw):
    monkeypatch.setattr(data, "load_fer2013", fake_fer2013)
    monkeypatch.setattr(data, "load_ckplus", functools.partial(data.load_ckplus, **ckplus_kw))
    return data
