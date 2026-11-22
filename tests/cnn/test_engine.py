import json

import pytest
import torch
from torch import nn

from src.cnn import engine
from src.cnn.datasets import make_loaders
from src.cnn.models import load_checkpoint
from src.cnn.train_resnet import main as main_resnet
from src.cnn.train_scratch import main as main_scratch
from src.common.metrics import validate_record


def _setup(lr, model):
    loaders = make_loaders("small", 32, 0, splits=("train", "val"))
    return loaders, torch.optim.Adam(model.parameters(), lr=lr), nn.CrossEntropyLoss()


def _linear():
    return nn.Sequential(nn.Flatten(), nn.Linear(2304, 7))


def test_fit_two_epochs_restores_best(fake_data):
    model = _linear()
    loaders, opt, crit = _setup(1e-2, model)
    r = engine.fit(model, loaders.train, loaders.val, crit, opt, "cpu", 2, 5, log=lambda *_: None)
    assert len(r["history"]) == 2
    _, _, loss, _ = engine.predict(model, loaders.val, "cpu", crit)
    assert abs(loss - r["best_val_loss"]) < 1e-5


def test_fit_early_stop(fake_data):
    model = _linear()
    loaders, opt, crit = _setup(0.0, model)  # lr=0: a perda de val nunca melhora
    r = engine.fit(model, loaders.train, loaders.val, crit, opt, "cpu", 10, 2, log=lambda *_: None)
    assert len(r["history"]) == 3 and r["best_epoch"] == 1


def test_fit_diverged_raises(fake_data):
    model = _linear()
    loaders, opt, _ = _setup(1e-2, model)
    nan_loss = lambda out, y: out.sum() * float("nan")  # noqa: E731
    with pytest.raises(RuntimeError, match="divergiu"):
        engine.fit(model, loaders.train, loaders.val, nan_loss, opt, "cpu", 2, 5, log=lambda *_: None)


def _check(tmp_path, name):
    rec = json.loads(next((tmp_path / "res").glob(f"{name}_*.json")).read_text())
    validate_record(rec)
    assert rec["split"] == "val"
    _, ckpt = load_checkpoint(tmp_path / "ck" / f"{name}.pt")
    assert ckpt["name"] == name


COMMON = ["--epochs", "1", "--limit-batches", "2", "--workers", "0", "--device", "cpu"]


def test_main_scratch(fake_data, tmp_path):
    main_scratch(COMMON + ["--ckpt-dir", str(tmp_path / "ck"), "--results-dir", str(tmp_path / "res")])
    _check(tmp_path, "cnn_scratch")


def test_main_resnet(fake_data, tmp_path):
    main_resnet(COMMON + ["--size", "112", "--no-pretrained", "--head-epochs", "1",
                          "--ckpt-dir", str(tmp_path / "ck"), "--results-dir", str(tmp_path / "res")])
    _check(tmp_path, "resnet18")
