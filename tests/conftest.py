"""Stubs TEMPORÁRIOS de ``src.common`` para os testes da Frente A.

Enquanto a Frente B não publica ``src/common/`` e os dados não estão nesta máquina,
os testes rodam com arrays aleatórios no formato do contrato (N x 48 x 48 uint8,
rótulos 0-6). Regras:

* Um módulo de ``src.common`` só é substituído se a importação real falhar. Quando a
  Frente B publicar ``degrade``, ``clean`` e ``metrics``, os testes passam a usar o
  código real automaticamente.
* ``load_fer2013`` e ``load_ckplus`` são SEMPRE trocados pelos falsos (fixture
  ``fake_data``), para os testes não dependerem dos dados baixados.

Nada aqui é usado fora dos testes. Apagar os stubs quando ``src/common`` existir.
"""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAKE_SIZES = {"train": 140, "val": 35, "test": 35}


# ---------------------------------------------------------------- dados falsos
def fake_fer2013(split: str):
    rng = np.random.default_rng({"train": 0, "val": 1, "test": 2}[split])
    n = FAKE_SIZES[split]
    images = rng.integers(0, 256, size=(n, 48, 48), dtype=np.uint8)
    labels = np.resize(np.arange(7), n)  # todas as classes presentes
    labels = labels[rng.permutation(n)].astype(np.int64)
    return images, labels


def fake_ckplus(fold: int, part: str = "train", mode: str = "sequence"):
    assert mode == "sequence"
    rng = np.random.default_rng(100 * fold + (part == "test"))
    n = 12 if part == "train" else 5
    lengths = rng.integers(4, 30, size=n)  # inclui sequências mais curtas que T=16
    sequences = [rng.integers(0, 256, size=(L, 64, 64), dtype=np.uint8) for L in lengths]
    labels = rng.integers(0, 7, size=n).astype(np.int64)
    return sequences, labels


# ------------------------------------------- stubs de módulos (só se faltarem)
def _stub_data() -> types.ModuleType:
    m = types.ModuleType("src.common.data")
    m.load_fer2013 = fake_fer2013
    m.load_ckplus = fake_ckplus
    return m


def _stub_degrade() -> types.ModuleType:
    from scipy import ndimage

    m = types.ModuleType("src.common.degrade")

    def add_gaussian_noise(img, sigma):
        noisy = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def gaussian_blur(img, ksize):
        return np.clip(ndimage.gaussian_filter(img.astype(np.float32), sigma=ksize / 6), 0, 255).astype(np.uint8)

    def downsample(img, size):
        small = ndimage.zoom(img.astype(np.float32), size / img.shape[0], order=1)
        back = ndimage.zoom(small, img.shape[0] / small.shape[0], order=1)
        return np.clip(back, 0, 255).astype(np.uint8)

    m.add_gaussian_noise, m.gaussian_blur, m.downsample = add_gaussian_noise, gaussian_blur, downsample
    return m


def _stub_clean() -> types.ModuleType:
    m = types.ModuleType("src.common.clean")

    def class_weights(labels):
        counts = np.bincount(np.asarray(labels), minlength=7).astype(np.float64)
        return counts.sum() / (7 * np.maximum(counts, 1))

    m.class_weights = class_weights
    return m


def _stub_metrics() -> types.ModuleType:
    m = types.ModuleType("src.common.metrics")

    def save_results(*, method, track, dataset, split, degradation, level, y_true, y_pred,
                     train_time_s, inference_ms_per_img, seed, out_dir="results"):
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        cm = np.zeros((7, 7), dtype=int)
        np.add.at(cm, (y_true, y_pred), 1)
        tp = np.diag(cm).astype(float)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
        record = {"method": method, "track": track, "dataset": dataset, "split": split,
                  "degradation": degradation, "level": level,
                  "accuracy": float((y_true == y_pred).mean()), "f1_macro": float(f1.mean()),
                  "f1_per_class": f1.tolist(), "confusion_matrix": cm.tolist(),
                  "train_time_s": train_time_s, "inference_ms_per_img": inference_ms_per_img,
                  "seed": seed, "git_commit": "stub"}
        path = Path(out_dir) / f"{method}_{dataset}_{split}_{degradation}_{level}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
        return path

    m.save_results = save_results
    return m


STUBS = {"data": _stub_data, "degrade": _stub_degrade, "clean": _stub_clean, "metrics": _stub_metrics}
STUBBED: list[str] = []


def _install_stubs() -> None:
    try:
        importlib.import_module("src.common")
    except ImportError:
        pkg = types.ModuleType("src.common")
        pkg.__path__ = []
        sys.modules["src.common"] = pkg
    for name, factory in STUBS.items():
        full = f"src.common.{name}"
        try:
            importlib.import_module(full)
        except ImportError:
            mod = factory()
            sys.modules[full] = mod
            setattr(sys.modules["src.common"], name, mod)
            STUBBED.append(name)


_install_stubs()


def pytest_report_header(config):
    return f"src.common com stub temporário: {', '.join(STUBBED) or 'nenhum'}"


@pytest.fixture
def fake_data(monkeypatch):
    """Troca os carregadores de dados reais pelos falsos (sempre)."""
    from src.common import data

    monkeypatch.setattr(data, "load_fer2013", fake_fer2013, raising=False)
    monkeypatch.setattr(data, "load_ckplus", fake_ckplus, raising=False)
    return data
