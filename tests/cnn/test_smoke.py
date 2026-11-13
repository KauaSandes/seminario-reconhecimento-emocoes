import json

import pytest

from scripts.smoke_test import run_smoke, save_smoke_json, bench_loader


@pytest.mark.parametrize("mode,size", [("small", None), ("resnet", 112)])  # 224 fica para o script (lento na CPU)
def test_tiny_cnn_memorizes_fixed_batch(fake_data, mode, size):
    r = run_smoke(mode, n=64, steps=60, size=size)
    assert r["accuracy"] >= 0.95, r
    assert r["last_loss"] < r["first_loss"]


def test_smoke_json_follows_contract(fake_data, tmp_path):
    r = run_smoke("small", n=64, steps=5)
    path = save_smoke_json(r, out_dir=tmp_path)
    rec = json.loads(open(path).read())
    expected = {"method", "track", "dataset", "split", "degradation", "level", "accuracy", "f1_macro",
                "f1_per_class", "confusion_matrix", "train_time_s", "inference_ms_per_img", "seed", "git_commit"}
    assert expected <= rec.keys()
    assert rec["method"] == "smoke" and rec["track"] == "cnn" and rec["split"] == "train"
    assert rec["degradation"] == "none" and rec["level"] == 0 and rec["seed"] == 42
    assert len(rec["f1_per_class"]) == 7 and len(rec["confusion_matrix"]) == 7


def test_bench_loader_runs(fake_data):
    rows = bench_loader("small", [0], batch_size=16, n_batches=3, size=None, seed=42)
    assert rows[0][0] == 0 and rows[0][1] > 0
