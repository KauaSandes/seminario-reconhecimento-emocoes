import pytest
import torch

from src.cnn.models import (SmallCNN, build_model, build_resnet18, load_checkpoint, save_checkpoint,
                            set_backbone_trainable)


def test_small_cnn_shape():
    assert SmallCNN()(torch.zeros(2, 1, 48, 48)).shape == (2, 7)


def test_resnet_shape_and_freeze():
    m = build_resnet18(pretrained=False)
    assert m.eval()(torch.zeros(2, 3, 112, 112)).shape == (2, 7)
    set_backbone_trainable(m, False)
    assert {n for n, p in m.named_parameters() if p.requires_grad} == {"fc.weight", "fc.bias"}
    set_backbone_trainable(m, True)
    assert all(p.requires_grad for p in m.parameters())


def test_build_model_invalid():
    with pytest.raises(ValueError):
        build_model("x")


def test_checkpoint_roundtrip(tmp_path):
    m = build_model("cnn_scratch").eval()
    save_checkpoint(tmp_path / "sub" / "a.pt", m, name="cnn_scratch", size=None, epoch=3, val_loss=1.5,
                    history=[{"epoch": 1, "val_loss": 1.5}], args={"lr": 1e-3, "dir": tmp_path})
    m2, ckpt = load_checkpoint(tmp_path / "sub" / "a.pt")
    x = torch.randn(2, 1, 48, 48)
    with torch.no_grad():
        assert torch.equal(m(x), m2(x))
    assert ckpt["mode"] == "small" and ckpt["epoch"] == 3 and not m2.training
