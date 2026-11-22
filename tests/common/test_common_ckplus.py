"""Testes do CK+ (src/common/data.py) com a pasta falsa tests/fixtures/ckplus_fake/."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.common import data


@pytest.fixture
def ck(ckplus_fake, tmp_path, monkeypatch):
    """Cache e folds montados a partir da pasta falsa, num diretório temporário."""
    monkeypatch.setattr(data, "detect_faces", lambda gray, **k: np.zeros((0, 4), dtype=np.int64))
    cache = data.build_ckplus_cache(ckplus_fake.root, tmp_path / "ckplus.npz")
    folds = data.make_ckplus_folds(cache, tmp_path / "ckplus_folds.csv")
    kw = dict(raw_dir=ckplus_fake.root, cache_path=cache, folds_csv=folds)
    return ckplus_fake, kw


def _expected(fake):
    """Sequências que devem entrar: com rótulo e sem desprezo."""
    return {k: data.CKPLUS_TO_FER[v] for k, v in fake.sequences.items() if v is not None and v != 2}


def test_label_mapping_and_discards(ck):
    fake, kw = ck
    expected = _expected(fake)
    folds = data.read_ckplus_folds(kw["folds_csv"])
    got = dict(zip(folds["subject"] + "/" + folds["sequence"], folds["label"]))
    assert got == expected
    assert "S103/001" not in got  # desprezo descartado
    assert "S106/001" not in got  # sem rótulo ignorada
    assert data.CKPLUS_TO_FER == {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}


def test_no_subject_in_train_and_test_of_same_fold(ck):
    _, kw = ck
    folds = data.read_ckplus_folds(kw["folds_csv"])
    assert sorted(folds["fold"].unique()) == list(range(10))
    for k in range(10):
        _, _, tr = data.load_ckplus(k, "train", "sequence", **kw)
        _, _, te = data.load_ckplus(k, "test", "sequence", **kw)
        subj_tr = {i.split("/")[0] for i in tr}
        subj_te = {i.split("/")[0] for i in te}
        assert subj_te and not subj_tr & subj_te
        assert len(tr) + len(te) == len(folds)
    # cada pessoa cai no teste de exatamente um fold
    assert (folds.groupby("subject")["fold"].nunique() == 1).all()


def test_image_mode(ck):
    fake, kw = ck
    X, y, ids = data.load_ckplus(0, "train", "image", **kw)
    assert X.dtype == np.uint8 and X.shape[1:] == (48, 48) and y.dtype == np.int64
    assert len(X) == len(y) == len(ids) and len(X) % 2 == 0
    peaks, neutrals = ids[0::2], ids[1::2]
    assert all(i.endswith("/peak") for i in peaks) and all(i.endswith("/neutral") for i in neutrals)
    assert (y[1::2] == data.NEUTRAL).all()
    expected = _expected(fake)
    assert [expected[i.rsplit("/", 1)[0]] for i in peaks] == y[0::2].tolist()


def test_sequence_mode_keeps_all_frames(ck):
    fake, kw = ck
    seqs, y, ids = data.load_ckplus(0, "test", "sequence", **kw)
    assert len(seqs) == len(y) == len(ids) > 0
    for s, key in zip(seqs, ids):
        assert s.dtype == np.uint8 and s.shape == (fake.n_frames[key], 48, 48)
    assert data.NEUTRAL not in y.tolist()  # no modo sequence não existe a classe neutro


def test_peak_and_neutral_are_last_and_first_frames(ck):
    _, kw = ck
    seqs, _, sids = data.load_ckplus(3, "test", "sequence", **kw)
    X, _, ids = data.load_ckplus(3, "test", "image", **kw)
    for j, key in enumerate(sids):
        assert ids[2 * j] == f"{key}/peak"
        assert np.array_equal(X[2 * j], seqs[j][-1]) and np.array_equal(X[2 * j + 1], seqs[j][0])


def test_center_fallback_when_no_face(ck):
    _, kw = ck
    c = np.load(kw["cache_path"])
    assert not c["face_detected"].any()
    assert (c["boxes"][:, 2] == 64).all() and (c["boxes"][:, 0] == 0).all()  # quadrado central de 64x64


def test_same_face_box_for_whole_sequence(ckplus_fake, tmp_path, monkeypatch):
    calls = []

    def fake_detector(gray, **kwargs):
        calls.append(gray.shape)
        return np.array([[8, 4, 30, 30], [0, 0, 40, 40]], dtype=np.int64)  # o maior é o segundo

    monkeypatch.setattr(data, "detect_faces", fake_detector)
    cache = data.build_ckplus_cache(ckplus_fake.root, tmp_path / "ck.npz")
    c = np.load(cache)
    n_seq = len(c["labels"])
    assert len(calls) == n_seq  # só o primeiro quadro de cada sequência passa pelo detector
    assert c["face_detected"].all() and (c["boxes"] == [0, 0, 40, 40]).all()
    # recorte coerente: o quadro 48x48 é o recorte [0:40, 0:40] redimensionado
    import cv2
    first = sorted((ckplus_fake.root / "cohn-kanade-images" / "S100" / "001").glob("S100_*.png"))[0]
    raw = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
    expected = cv2.resize(raw[0:40, 0:40], (48, 48), interpolation=cv2.INTER_AREA)
    assert np.array_equal(c["frames"][0], expected)


def test_color_frames_become_gray(ck):
    _, kw = ck
    """A pessoa S101 tem quadros coloridos (BGR); depois do cache eles são [T, 48, 48] em cinza."""
    seqs, _, ids = data.load_ckplus(0, "train", "sequence", **kw)
    seqs_te, _, ids_te = data.load_ckplus(0, "test", "sequence", **kw)
    colored = [s for s, i in zip(list(seqs) + list(seqs_te), list(ids) + list(ids_te)) if i.startswith("S101/")]
    assert colored and all(s.ndim == 3 and s.shape[1:] == (48, 48) for s in colored)


def test_errors(ck, tmp_path):
    _, kw = ck
    with pytest.raises(ValueError):
        data.load_ckplus(10, "train", **kw)
    with pytest.raises(ValueError):
        data.load_ckplus(0, "val", **kw)
    with pytest.raises(ValueError):
        data.load_ckplus(0, "train", "video", **kw)
    with pytest.raises(FileNotFoundError, match="LEIA-ME"):
        data.load_ckplus(0, raw_dir=tmp_path / "vazio", cache_path=tmp_path / "x.npz", folds_csv=kw["folds_csv"])
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        data.load_ckplus(0, cache_path=kw["cache_path"], folds_csv=tmp_path / "nao.csv")


def test_folds_file_mismatch_is_detected(ck, tmp_path):
    _, kw = ck
    df = pd.read_csv(kw["folds_csv"], dtype=str)
    bad = tmp_path / "bad.csv"
    df.iloc[1:].to_csv(bad, index=False)
    with pytest.raises(ValueError, match="não corresponde"):
        data.load_ckplus(0, **{**kw, "folds_csv": bad})
