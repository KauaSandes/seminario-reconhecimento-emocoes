"""Testes do scripts/download_data.py (sem rede e sem credenciais reais)."""
from __future__ import annotations

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.common.data import read_fer2013_csv

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("download_data", ROOT / "scripts" / "download_data.py")
dd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dd)


# ------------------------------------------------------------------ validação
def test_validator_accepts_synthetic_csv(fer_csv):
    df, X = read_fer2013_csv(fer_csv.path, fer_csv.sizes)
    assert list(df.columns) == ["emotion", "pixels", "Usage"]
    assert X.shape == (sum(fer_csv.sizes.values()), 48, 48) and X.dtype.name == "uint8"


def test_validator_rejects_official_counts_on_small_csv(fer_csv):
    with pytest.raises(ValueError, match="esperado 35887"):
        read_fer2013_csv(fer_csv.path)  # contagens oficiais por padrão


def test_validator_rejects_missing_column(fer_csv, tmp_path):
    bad = tmp_path / "bad.csv"
    pd.read_csv(fer_csv.path).drop(columns=["Usage"]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="colunas"):
        read_fer2013_csv(bad, fer_csv.sizes)


def test_validator_rejects_wrong_pixel_count(fer_csv, tmp_path):
    df = pd.read_csv(fer_csv.path, dtype={"pixels": str})
    df.loc[7, "pixels"] = " ".join(df.loc[7, "pixels"].split()[:2303])
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="2304"):
        read_fer2013_csv(bad, fer_csv.sizes)


def test_validator_rejects_out_of_range_pixel(fer_csv, tmp_path):
    df = pd.read_csv(fer_csv.path, dtype={"pixels": str})
    df.loc[0, "pixels"] = "256 " + " ".join(df.loc[0, "pixels"].split()[1:])
    bad = tmp_path / "bad.csv"
    df.to_csv(bad, index=False)
    with pytest.raises(ValueError, match="0..255"):
        read_fer2013_csv(bad, fer_csv.sizes)


# ----------------------------------------------------------------- credenciais
def test_credentials_detection(tmp_path):
    assert dd.find_kaggle_credentials(env={}, home=tmp_path) is None
    assert dd.find_kaggle_credentials(env={"KAGGLE_USERNAME": "u"}, home=tmp_path) is None
    assert "KAGGLE_KEY" in dd.find_kaggle_credentials(env={"KAGGLE_USERNAME": "u", "KAGGLE_KEY": "k"}, home=tmp_path)
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text("{}")
    assert dd.find_kaggle_credentials(env={}, home=tmp_path).endswith("kaggle.json")


def test_stops_without_credentials(tmp_path, monkeypatch, capsys):
    for var in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN", "KAGGLE_CONFIG_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(dd.Path, "home", classmethod(lambda cls: tmp_path))

    def no_network(*a, **k):
        raise AssertionError("não deveria chamar o kaggle sem credencial")

    monkeypatch.setattr(dd.subprocess, "run", no_network)
    code = dd.main(["--only", "fer2013", "--fer-dir", str(tmp_path / "fer")])
    err = capsys.readouterr().err
    assert code == 1
    assert "aceitar as regras" in err and "kaggle.json" in err


def test_auth_error_detection():
    assert dd.AUTH_ERROR.search("403 - Forbidden - You must accept this competition's rules")
    assert dd.AUTH_ERROR.search("401 Client Error: Unauthorized for url")
    assert not dd.AUTH_ERROR.search("100%|██████████| 285M/285M [00:09<00:00, 40.1MB/s]")


# ------------------------------------------------------------------- extração
def _make_competition_zip(zip_path: Path, csv_path: Path, use_tar: bool) -> None:
    with zipfile.ZipFile(zip_path, "w") as zf:
        if use_tar:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                tf.add(csv_path, arcname="fer2013/fer2013.csv")
            zf.writestr("fer2013.tar.gz", buf.getvalue())
        else:  # icml_face_data.csv: outra ordem de colunas e espaço no começo dos nomes
            df = pd.read_csv(csv_path, dtype={"pixels": str})
            df = df[["emotion", "Usage", "pixels"]].rename(columns={"Usage": " Usage", "pixels": " pixels"})
            zf.writestr("icml_face_data.csv", df.to_csv(index=False, lineterminator="\n"))
        zf.writestr("example_submission.csv", "id\n")


@pytest.mark.parametrize("use_tar", [True, False])
def test_extract_from_competition_zip(fer_csv, tmp_path, use_tar):
    zip_path = tmp_path / f"{dd.COMPETITION}.zip"
    _make_competition_zip(zip_path, fer_csv.path, use_tar)
    out = tmp_path / "out" / "fer2013.csv"
    source = dd.extract_fer2013(zip_path, out)
    assert ("tar.gz" in source) == use_tar
    df, X = read_fer2013_csv(out, fer_csv.sizes)
    ref, X_ref = read_fer2013_csv(fer_csv.path, fer_csv.sizes)
    assert (X == X_ref).all() and df.equals(ref)
    if use_tar:  # bytes idênticos ao original
        assert dd.sha256_file(out) == dd.sha256_file(fer_csv.path)


def test_idempotent_when_zip_already_present(fer_csv, tmp_path, monkeypatch):
    """Com o zip já baixado, não chama o kaggle. Com o CSV já extraído, não faz nada."""
    fer_dir = tmp_path / "fer"
    fer_dir.mkdir()
    _make_competition_zip(fer_dir / f"{dd.COMPETITION}.zip", fer_csv.path, use_tar=True)
    monkeypatch.setattr(dd, "FER2013_SIZES", fer_csv.sizes)
    monkeypatch.setattr(dd.subprocess, "run", lambda *a, **k: pytest.fail("não deveria baixar"))
    csv = dd.prepare_fer2013(fer_dir)
    checksum = (fer_dir / "CHECKSUM.txt").read_text()
    assert dd.sha256_file(csv) in checksum
    mtime = csv.stat().st_mtime_ns
    dd.prepare_fer2013(fer_dir)  # segunda vez: só valida
    assert csv.stat().st_mtime_ns == mtime


# ------------------------------------------------------------------------- CK+
def test_ckplus_without_zips_warns_and_succeeds(tmp_path, capsys):
    assert dd.prepare_ckplus(tmp_path / "ck") is False
    assert (tmp_path / "ck" / "LEIA-ME.txt").is_file()
    assert "Seguindo sem o CK+" in capsys.readouterr().out


def test_ckplus_extracts_nested_zips(ckplus_fake, tmp_path):
    ck = tmp_path / "ck"
    ck.mkdir()
    for zip_name, folder in (("extended-cohn-kanade-images.zip", "cohn-kanade-images"),
                             ("Emotion_labels.zip", "Emotion")):
        with zipfile.ZipFile(ck / zip_name, "w") as zf:
            for f in (ckplus_fake.root / folder).rglob("*"):
                if f.is_file():
                    # Emotion vem aninhado em Emotion_labels/, como em algumas distribuições
                    prefix = "Emotion_labels/" if folder == "Emotion" else ""
                    zf.write(f, prefix + f.relative_to(ckplus_fake.root).as_posix())
            zf.writestr("__MACOSX/._lixo", "x")
    assert dd.prepare_ckplus(ck) is True
    assert (ck / "Emotion_labels" / "Emotion").is_dir()
    assert not (ck / "__MACOSX").exists()
