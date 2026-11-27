"""Testes da Frente B (src/classic) com o CSV sintético de tests/common/conftest.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.classic import features, preprocess, train_classic
from src.common import clean, data
from src.common.metrics import validate_record

_spec = importlib.util.spec_from_file_location("common_conftest", Path(__file__).parents[1] / "common" / "conftest.py")
_conf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _conf   # as dataclasses do conftest precisam do módulo registrado
_spec.loader.exec_module(_conf)


@pytest.fixture
def fer(tmp_path):
    fx = _conf.make_fer_csv(tmp_path / "fer2013.csv")
    cache = tmp_path / "fer.npz"
    arr = data.fer2013_arrays(fx.path, cache)
    dec = clean.clean_fer2013(arr["X"], arr["y"], arr["split"], arr["ids"], face_check=False)
    splits = clean.write_clean_csv(dec, tmp_path / "clean.csv")
    return ["--csv-path", str(fx.path), "--cache-path", str(cache), "--splits-csv", str(splits)]


def test_preprocess_keeps_shape_and_type():
    X = np.random.default_rng(0).integers(0, 256, (5, 48, 48), dtype=np.uint8)
    out = preprocess.preprocess(X)
    assert out.shape == X.shape and out.dtype == np.uint8
    with pytest.raises(ValueError):
        preprocess.preprocess(X.astype(np.float32))


def test_feature_shapes():
    X = np.random.default_rng(0).integers(0, 256, (4, 48, 48), dtype=np.uint8)
    assert features.hog_features(X).shape == (4, 900)
    px = features.pixel_features(X)
    assert px.shape == (4, 2304) and px.dtype == np.float32 and px.max() <= 1.0


@pytest.mark.parametrize("extra,method", [(["--model", "majority"], "majority"),
                                          (["--features", "hog", "--C", "0.1"], "hog_linearsvc"),
                                          (["--features", "pixels", "--C", "0.1"], "pixels_linearsvc")])
def test_end_to_end_writes_val_json_and_never_loads_test(fer, tmp_path, monkeypatch, extra, method):
    loaded = []
    real = data.load_fer2013

    def spy(split, *a, **k):
        loaded.append(split)
        return real(split, *a, **k)

    monkeypatch.setattr(data, "load_fer2013", spy)
    out = tmp_path / "results"
    assert train_classic.main([*fer, *extra, "--results-dir", str(out)]) == 0
    assert "test" not in loaded and set(loaded) == {"train", "val"}
    files = list(out.glob("*.json"))
    assert [f.name for f in files] == [f"{method}_fer2013_val_none_0.json"]
    rec = json.loads(files[0].read_text(encoding="utf-8"))
    validate_record(rec)
    assert rec["split"] == "val"
