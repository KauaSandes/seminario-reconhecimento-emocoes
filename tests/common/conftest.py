"""Dados sintéticos para os testes de ``src.common`` (sem dados reais nem rede).

- ``fer_csv``: um ``fer2013.csv`` pequeno, no mesmo formato do oficial, com problemas
  plantados (duplicatas, vazamento entre conjuntos, rótulos conflitantes e imagens
  constantes).
- ``ckplus_fake``: pasta falsa do CK+ em ``tests/fixtures/ckplus_fake/``, gerada de
  forma determinística a cada sessão (fica fora do git).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FER_SIZES = {"Training": 40, "PublicTest": 12, "PrivateTest": 12}


# --------------------------------------------------------------- FER sintética
@dataclass
class FerFixture:
    path: Path
    sizes: dict[str, int]
    planted: dict[str, list[int]] = field(default_factory=dict)  # motivo -> ids (linha do CSV)


def _random_face(rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 256, size=(48, 48), dtype=np.uint8)


def make_fer_csv(path: Path, seed: int = 0) -> FerFixture:
    """Cria o CSV sintético. Os ids são as linhas: 0..39 treino, 40..51 val e 52..63 teste."""
    rng = np.random.default_rng(seed)
    usages = (["Training"] * FER_SIZES["Training"] + ["PublicTest"] * FER_SIZES["PublicTest"]
              + ["PrivateTest"] * FER_SIZES["PrivateTest"])
    n = len(usages)
    images = [_random_face(rng) for _ in range(n)]
    labels = list(np.resize(np.arange(7), n))
    tr, va, te = 0, FER_SIZES["Training"], FER_SIZES["Training"] + FER_SIZES["PublicTest"]

    # (a) duplicata dentro do treino, com o mesmo rótulo: fica o id 0 e sai o id 1
    images[1], labels[1] = images[0].copy(), labels[0]
    # (b) vazamento: o treino 2 é igual ao val 40 (mesmo rótulo), então o treino 2 sai
    images[2], labels[2] = images[va].copy(), labels[va]
    # (c) conflito dentro do treino: 3 e 4 iguais com rótulos diferentes, e os dois saem
    images[4], labels[4] = images[3].copy(), (labels[3] + 1) % 7
    # (c') conflito entre treino e teste: o treino 5 sai e o teste 52 só é sinalizado
    images[5], labels[5] = images[te].copy(), (labels[te] + 2) % 7
    # duplicata dentro do val (41 = 42, mesmo rótulo): só sinaliza
    images[va + 2], labels[va + 2] = images[va + 1].copy(), labels[va + 1]
    # quase constantes: treino 6 (sai) e val 43 (só sinaliza); valores diferentes para não serem duplicatas
    images[6] = np.full((48, 48), 10, np.uint8)
    images[6][0, 0] = 12
    images[va + 3] = np.full((48, 48), 200, np.uint8)

    df = pd.DataFrame({
        "emotion": labels,
        "pixels": [" ".join(map(str, img.ravel())) for img in images],
        "Usage": usages,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    planted = {
        "dup_within_split": [1],
        "dup_cross_split_leak": [2],
        "label_conflict": [3, 4, 5],
        "low_std": [6],
        "flag:label_conflict": [te],
        "flag:dup_within_split": [va + 1, va + 2],
        "flag:low_std": [va + 3],
    }
    return FerFixture(path=path, sizes=dict(FER_SIZES), planted=planted)


@pytest.fixture
def fer_csv(tmp_path: Path) -> FerFixture:
    return make_fer_csv(tmp_path / "raw" / "fer2013" / "fer2013.csv")


# ------------------------------------------------------------ CK+ falso
@dataclass
class CkFixture:
    root: Path
    sequences: dict[str, int | None]  # "Sxxx/yyy" -> rótulo original do CK+ (None = sem rótulo)
    n_frames: dict[str, int]


def _ck_frame(rng: np.random.Generator, size: int = 64, color: bool = False) -> np.ndarray:
    img = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    if color:
        return np.stack([img, img // 2, 255 - img], axis=-1)
    return img


def make_ckplus_fake(root: Path, seed: int = 0) -> CkFixture:
    """12 pessoas com 1 ou 2 sequências de 3 a 6 quadros, mais casos de borda.

    Um rótulo 2 (desprezo, descartado), uma sequência sem rótulo (ignorada), uma
    sequência colorida e arquivos .DS_Store e ._* (lixo de macOS) que devem ser ignorados.
    """
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(seed)
    images_root = root / "cohn-kanade-images"
    emotion_root = root / "Emotion"
    labels_cycle = [1, 3, 4, 5, 6, 7]  # todos os rótulos válidos (sem o desprezo)
    sequences: dict[str, int | None] = {}
    n_frames: dict[str, int] = {}
    k = 0
    for s in range(12):
        subject = f"S{100 + s:03d}"
        for q in range(1 + s % 2):
            seq = f"{q + 1:03d}"
            key = f"{subject}/{seq}"
            label: int | None = labels_cycle[k % len(labels_cycle)]
            k += 1
            if subject == "S103" and seq == "001":
                label = 2  # desprezo
            if subject == "S106" and seq == "001":
                label = None  # sem rótulo
            T = 3 + (k % 4)
            seq_dir = images_root / subject / seq
            seq_dir.mkdir(parents=True, exist_ok=True)
            for t in range(T):
                frame = _ck_frame(rng, color=(subject == "S101"))
                cv2.imwrite(str(seq_dir / f"{subject}_{seq}_{t + 1:08d}.png"), frame)
            (seq_dir / ".DS_Store").write_bytes(b"\x00")
            (seq_dir / f"._{subject}_{seq}_00000001.png").write_bytes(b"\x00")
            if label is not None:
                emo_dir = emotion_root / subject / seq
                emo_dir.mkdir(parents=True, exist_ok=True)
                (emo_dir / f"{subject}_{seq}_{T:08d}_emotion.txt").write_text(f"   {float(label):.7e}\n")
            sequences[key] = label
            n_frames[key] = T
    return CkFixture(root=root, sequences=sequences, n_frames=n_frames)


@pytest.fixture(scope="session")
def ckplus_fake() -> CkFixture:
    return make_ckplus_fake(FIXTURES / "ckplus_fake")
