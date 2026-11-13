"""Testes do carregamento da FER-2013 (src/common/data.py) com o CSV sintético."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.common import data


@pytest.fixture
def paths(fer_csv, tmp_path):
    return {"csv_path": fer_csv.path, "cache_path": tmp_path / "processed" / "fer2013.npz"}


def _write_keep_all(arr, path):
    pd.DataFrame({"id": arr["ids"], "split": arr["split"], "label": arr["y"],
                  "keep": True, "reason": ""}).to_csv(path, index=False)


def test_label_order_is_official():
    assert data.EMOTIONS == ("raiva", "nojo", "medo", "alegria", "tristeza", "surpresa", "neutro")
    assert data.USAGE_TO_SPLIT == {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}


def test_arrays_shapes_and_types(fer_csv, paths):
    arr = data.fer2013_arrays(**paths)
    n = sum(fer_csv.sizes.values())
    assert arr["X"].shape == (n, 48, 48) and arr["X"].dtype == np.uint8
    assert arr["y"].dtype == np.int64 and arr["y"].min() >= 0 and arr["y"].max() <= 6
    assert np.array_equal(arr["ids"], np.arange(n))
    assert {s: int((arr["split"] == s).sum()) for s in data.SPLITS} == {"train": 40, "val": 12, "test": 12}


def test_pixels_follow_row_major_order(fer_csv, paths):
    row = pd.read_csv(fer_csv.path, dtype={"pixels": str}).loc[10, "pixels"]
    expected = np.array(row.split(), dtype=np.uint8).reshape(48, 48)
    assert np.array_equal(data.fer2013_arrays(**paths)["X"][10], expected)


def test_cache_is_created_and_reused(paths):
    a = data.fer2013_arrays(**paths)
    assert paths["cache_path"].is_file()
    mtime = paths["cache_path"].stat().st_mtime_ns
    b = data.fer2013_arrays(**paths)
    assert paths["cache_path"].stat().st_mtime_ns == mtime
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_cache_is_rebuilt_when_csv_changes(fer_csv, paths):
    data.fer2013_arrays(**paths)
    df = pd.read_csv(fer_csv.path, dtype={"pixels": str})
    df.loc[0, "emotion"] = (df.loc[0, "emotion"] + 1) % 7
    df.to_csv(fer_csv.path, index=False, lineterminator="\n")
    st = fer_csv.path.stat()
    os.utime(fer_csv.path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert data.fer2013_arrays(**paths)["y"][0] == df.loc[0, "emotion"]


@pytest.mark.parametrize("split,n", [("train", 40), ("val", 12), ("test", 12)])
def test_load_unclean(paths, split, n):
    X, y, ids = data.load_fer2013(split, clean=False, **paths)
    assert X.shape == (n, 48, 48) and X.dtype == np.uint8
    assert y.shape == (n,) and y.dtype == np.int64
    assert ids.shape == (n,) and len(np.unique(ids)) == n


def test_load_clean_applies_keep(paths, tmp_path):
    arr = data.fer2013_arrays(**paths)
    splits_csv = tmp_path / "clean.csv"
    _write_keep_all(arr, splits_csv)
    df = pd.read_csv(splits_csv)
    df.loc[[0, 1], "keep"] = False
    df.to_csv(splits_csv, index=False)
    X, y, ids = data.load_fer2013("train", splits_csv=splits_csv, **paths)
    assert len(X) == 38 and 0 not in ids and 1 not in ids
    assert np.array_equal(y, arr["y"][ids])


def test_load_clean_requires_splits_file(paths, tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        data.load_fer2013("train", splits_csv=tmp_path / "missing.csv", **paths)


def test_load_clean_rejects_mismatched_file(paths, tmp_path):
    arr = data.fer2013_arrays(**paths)
    splits_csv = tmp_path / "clean.csv"
    _write_keep_all(arr, splits_csv)
    df = pd.read_csv(splits_csv)
    df.loc[0, "label"] = (df.loc[0, "label"] + 1) % 7
    df.to_csv(splits_csv, index=False)
    with pytest.raises(ValueError, match="não corresponde"):
        data.load_fer2013("val", splits_csv=splits_csv, **paths)


def test_invalid_split(paths):
    with pytest.raises(ValueError):
        data.load_fer2013("validation", clean=False, **paths)


def test_missing_csv(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        data.load_fer2013("train", csv_path=tmp_path / "nada.csv", cache_path=tmp_path / "c.npz")
