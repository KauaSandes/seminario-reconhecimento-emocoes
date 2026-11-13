import numpy as np
import pytest
import torch

from src.cnn import video_data as V


@pytest.mark.parametrize("n,T", [(30, 16), (16, 16), (5, 16), (1, 4)])
def test_sample_frame_indices(n, T):
    idx = V.sample_frame_indices(n, T)
    assert len(idx) == T
    assert idx[0] == 0 and idx[-1] == n - 1  # do neutro ao pico
    assert np.all(np.diff(idx) >= 0)
    if n < T:
        assert list(idx[:n]) == list(range(n)) and np.all(idx[n:] == n - 1)


@pytest.mark.parametrize("mode,shape", [("small", (16, 1, 48, 48)), ("resnet", (16, 3, 112, 112))])
def test_ck_sequence_dataset(fake_data, mode, shape):
    ds = V.CKSequenceDataset(fold=0, part="train", T=16, mode=mode, size=112 if mode == "resnet" else None)
    seqs, labels, ids = fake_data.load_ckplus(0, part="train", mode="sequence")
    assert len(ds) == len(seqs) and list(ds.ids) == list(ids)
    assert any(len(s) < 16 for s in seqs)  # o CK+ falso tem 3 a 6 quadros: testa a repetição
    for i in range(len(ds)):
        x, y = ds[i]
        assert x.shape == shape and y.item() == labels[i]


def test_short_sequence_repeats_last_frame(fake_data):
    seq = np.stack([np.full((48, 48), v, np.uint8) for v in (0, 100, 200)])
    ds = V.CKSequenceDataset(0, "train", T=6, sequences=[seq], labels=[2])
    x, _ = ds[0]
    assert torch.allclose(x[:, 0, 0, 0], torch.tensor([0, 100, 200, 200, 200, 200]) / 255)


def test_extract_features(fake_data, tmp_path):
    torch.manual_seed(0)
    backbone = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())
    path = V.extract_features(backbone, fold=3, model_name="tiny", T=8, out_dir=tmp_path)
    assert path.name == "ck_features_tiny_fold3.pt"
    data = torch.load(path)
    for part in ("train", "test"):
        f, y = data[part]["features"], data[part]["labels"]
        assert f.shape == (len(y), 8, 4)
    assert data["T"] == 8 and data["model"] == "tiny"
    assert not any(p.requires_grad for p in backbone.parameters())


def test_extract_features_resnet18(fake_data, tmp_path):
    import torchvision
    m = torchvision.models.resnet18(weights=None)
    m.fc = torch.nn.Identity()
    path = V.extract_features(m, fold=0, model_name="resnet18", T=4, mode="resnet", size=112, out_dir=tmp_path)
    assert torch.load(path)["train"]["features"].shape[-1] == 512
