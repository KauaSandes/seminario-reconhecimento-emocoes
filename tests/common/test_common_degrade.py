"""Testes das degradações (src/common/degrade.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.common import degrade


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(8, 48, 48), dtype=np.uint8)


ALL_LEVELS = [(k, lv) for k, lvs in degrade.LEVELS.items() for lv in lvs]


@pytest.mark.parametrize("kind,level", ALL_LEVELS)
def test_shape_and_dtype_preserved(batch, kind, level):
    out = degrade.apply_degradation(batch, kind, level)
    assert out.shape == batch.shape and out.dtype == np.uint8
    assert out is not batch


def test_levels_match_contract():
    assert degrade.LEVELS == {"none": (0,), "noise": (5, 10, 20), "blur": (3, 5, 7), "resolution": (24, 12)}


def test_none_is_identity(batch):
    assert np.array_equal(degrade.apply_degradation(batch, "none", 0), batch)


def test_noise_is_reproducible_and_varies_between_images():
    X = np.full((4, 48, 48), 128, dtype=np.uint8)
    a = degrade.apply_degradation(X, "noise", 10, seed=42)
    b = degrade.apply_degradation(X, "noise", 10, seed=42)
    c = degrade.apply_degradation(X, "noise", 10, seed=7)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert not np.array_equal(a[0], a[1])  # um único gerador para o lote: ruído diferente por imagem


@pytest.mark.parametrize("sigma", [5, 10, 20])
def test_noise_std_close_to_sigma(sigma):
    img = np.full((256, 256), 128, dtype=np.uint8)
    diff = degrade.add_gaussian_noise(img, sigma).astype(np.float64) - 128
    assert abs(diff.std() - sigma) < 0.05 * sigma + 0.3
    assert abs(diff.mean()) < 0.2


def test_noise_clips_to_valid_range():
    img = np.array([[0, 255]], dtype=np.uint8)
    out = degrade.add_gaussian_noise(np.tile(img, (100, 100)), 50)
    assert out.min() >= 0 and out.max() <= 255 and out.dtype == np.uint8


def test_single_image_noise_default_seed_is_fixed(batch):
    assert np.array_equal(degrade.add_gaussian_noise(batch[0], 10), degrade.add_gaussian_noise(batch[0], 10))


@pytest.mark.parametrize("ksize", [3, 5, 7])
def test_blur_reduces_high_frequencies(batch, ksize):
    out = degrade.gaussian_blur(batch[0], ksize)
    assert np.abs(np.diff(out.astype(int), axis=1)).mean() < np.abs(np.diff(batch[0].astype(int), axis=1)).mean()
    flat = np.full((48, 48), 77, dtype=np.uint8)
    assert np.array_equal(degrade.gaussian_blur(flat, ksize), flat)  # imagem constante não muda


def test_downsample_limits_detail():
    x = np.indices((48, 48)).sum(axis=0) % 2 * 255  # tabuleiro: frequência máxima
    x = x.astype(np.uint8)
    out = degrade.downsample(x, 24)
    assert out.shape == (48, 48)
    assert out.std() < 0.1 * x.std()  # o antialias elimina o padrão em vez de criar aliasing
    flat = np.full((48, 48), 200, dtype=np.uint8)
    assert np.array_equal(degrade.downsample(flat, 12), flat)


def test_invalid_inputs(batch):
    with pytest.raises(ValueError):
        degrade.apply_degradation(batch, "noise", 7)
    with pytest.raises(ValueError):
        degrade.apply_degradation(batch, "jpeg", 10)
    with pytest.raises(ValueError):
        degrade.gaussian_blur(batch[0], 4)
    with pytest.raises(ValueError):
        degrade.downsample(batch[0], 48)
    with pytest.raises(TypeError):
        degrade.apply_degradation(batch.astype(np.float32), "blur", 3)
