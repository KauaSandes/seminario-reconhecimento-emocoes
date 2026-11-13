import numpy as np
import pytest
import torch

from src.cnn import datasets as D
from tests.conftest import FAKE_SIZES, STUBBED, fake_fer2013


def test_small_mode_shape_and_range(fake_data):
    ds = D.FERDataset("val", mode="small")
    x, y = ds[0]
    assert x.shape == (1, 48, 48) and x.dtype == torch.float32
    assert 0.0 <= x.min() and x.max() <= 1.0
    assert y.dtype == torch.long and 0 <= y < 7
    assert len(ds) == FAKE_SIZES["val"]


def test_small_mode_preserves_pixels(fake_data):
    images, labels = fake_fer2013("val")
    x, y = D.FERDataset("val", mode="small")[3]
    assert torch.allclose(x[0], torch.from_numpy(images[3]).float() / 255)
    assert y.item() == labels[3]


@pytest.mark.parametrize("size", [224, 112])
def test_resnet_mode(fake_data, size):
    x, _ = D.FERDataset("val", mode="resnet", size=size)[0]
    assert x.shape == (3, size, size)
    # Os 3 canais vêm do mesmo cinza: desfazendo a normalização ficam iguais.
    raw = x * torch.tensor(D.IMAGENET_STD)[:, None, None] + torch.tensor(D.IMAGENET_MEAN)[:, None, None]
    assert torch.allclose(raw[0], raw[1], atol=1e-5) and torch.allclose(raw[0], raw[2], atol=1e-5)


def test_augment_only_in_train(fake_data):
    with pytest.raises(ValueError):
        D.FERDataset("val", augment=True)
    ds = D.FERDataset("train", mode="small", augment=True)
    torch.manual_seed(0)
    a = torch.stack([ds[0][0] for _ in range(8)])
    assert a.shape == (8, 1, 48, 48)
    assert not all(torch.equal(a[0], a[i]) for i in range(1, 8)), "aumento deveria variar a imagem"


def test_no_augment_is_deterministic(fake_data):
    ds = D.FERDataset("train", mode="resnet", size=112, augment=False)
    assert torch.equal(ds[5][0], ds[5][0])


def test_degradation_only_on_test(fake_data):
    with pytest.raises(ValueError):
        D.FERDataset("val", degradation="noise", level=10)
    with pytest.raises(ValueError):
        D.FERDataset("test", degradation="jpeg", level=10)


@pytest.mark.parametrize("name,level", [("noise", 10), ("blur", 5), ("resolution", 12)])
def test_degradation_uses_common_before_transform(fake_data, name, level):
    from src.common import degrade

    images, _ = fake_fer2013("test")
    fn = getattr(degrade, D.DEGRADATIONS[name])
    np.random.seed(42)
    expected = fn(images[0], level)

    ds = D.FERDataset("test", mode="small", degradation=name, level=level, seed=42)
    assert np.array_equal(ds.images[0], expected), "degradação deve ser a de src.common, aplicada no uint8"
    assert torch.allclose(ds[0][0][0], torch.from_numpy(expected).float() / 255)
    # Mesma semente -> mesmo ruído
    ds2 = D.FERDataset("test", mode="small", degradation=name, level=level, seed=42)
    assert np.array_equal(ds.images, ds2.images)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        D.FERDataset("train", images=np.zeros((2, 48, 48), np.float32), labels=np.zeros(2))
    with pytest.raises(ValueError):
        D.FERDataset("train", images=np.zeros((2, 48, 48), np.uint8), labels=np.array([0, 7]))
    with pytest.raises(ValueError):
        D.FERDataset("train", mode="vgg", images=np.zeros((1, 48, 48), np.uint8), labels=np.zeros(1))


@pytest.mark.parametrize("mode,shape", [("small", (16, 1, 48, 48)), ("resnet", (16, 3, 112, 112))])
def test_make_loaders_shapes(fake_data, mode, shape):
    loaders = D.make_loaders(mode=mode, batch_size=16, num_workers=0, size=112 if mode == "resnet" else None)
    x, y = next(iter(loaders.train))
    assert x.shape == shape and y.shape == (16,)
    assert not loaders.val.dataset.augment and not loaders.test.dataset.augment
    assert loaders.class_weights is None


def test_make_loaders_reproducible(fake_data):
    def first_batch():
        loaders = D.make_loaders("small", batch_size=8, num_workers=0, seed=42, splits=("train",))
        return next(iter(loaders.train))
    torch.manual_seed(0)
    xa, ya = first_batch()
    torch.manual_seed(123)  # estado global diferente não deve importar
    xb, yb = first_batch()
    assert torch.equal(ya, yb)


def test_make_loaders_splits_skip_test(fake_data, monkeypatch):
    calls = []
    orig = fake_data.load_fer2013
    monkeypatch.setattr(fake_data, "load_fer2013", lambda s: calls.append(s) or orig(s))
    loaders = D.make_loaders("small", num_workers=0, splits=("train", "val"))
    assert loaders.test is None and "test" not in calls


def test_balance_loss_and_sampler(fake_data, monkeypatch):
    # Treino bem desbalanceado: 120 da classe 3, 20 espalhados.
    images = np.zeros((140, 48, 48), np.uint8)
    labels = np.array([3] * 120 + list(range(7)) * 2 + [0] * 6)
    monkeypatch.setattr(fake_data, "load_fer2013",
                        lambda s: (images, labels) if s == "train" else fake_fer2013(s))

    loss_l = D.make_loaders("small", batch_size=20, num_workers=0, balance="loss", splits=("train",))
    w = loss_l.class_weights
    assert w.shape == (7,) and w[3] < w[1]

    samp = D.make_loaders("small", batch_size=140, num_workers=0, balance="sampler", splits=("train",))
    assert samp.class_weights is None
    assert isinstance(samp.train.sampler, torch.utils.data.WeightedRandomSampler)
    ys = torch.cat([y for _, y in samp.train] + [y for _, y in samp.train])
    assert (ys == 3).float().mean() < 0.5, "sampler deveria reduzir a classe majoritária (86% sem ele)"


@pytest.mark.skipif(bool(STUBBED), reason="workers no Windows (spawn) reimportam src.common e não "
                    "enxergam o stub; roda quando o src/common da Frente B existir")
def test_workers_seeded(fake_data):
    # Com src/common real os workers ainda carregam os arrays falsos já guardados no dataset.
    loaders = D.make_loaders("small", batch_size=8, num_workers=2, splits=("train",), augment=True)
    x, y = next(iter(loaders.train))
    assert x.shape == (8, 1, 48, 48)
