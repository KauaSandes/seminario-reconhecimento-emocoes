import numpy as np
import pytest
import torch
from torch import nn

from src.cnn.webcam_demo import draw_predictions, predict_frame, preprocess_face
from src.common.data import EMOTIONS


@pytest.fixture
def frame():
    return np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)


@pytest.mark.parametrize("box", [(10, 10, 60, 60), (130, 90, 60, 60)])  # a segunda passa da borda
def test_preprocess_face_shapes(frame, box):
    gray = frame[..., 0]
    assert preprocess_face(gray, box, "small", None).shape == (1, 1, 48, 48)
    assert preprocess_face(gray, box, "resnet", 112).shape == (1, 3, 112, 112)


def test_predict_frame(frame):
    model = nn.Sequential(nn.Flatten(), nn.Linear(48 * 48, 7)).eval()
    preds = predict_frame(frame, model, "small", None, "cpu", detector=lambda g: np.array([[10, 10, 40, 40]]))
    assert len(preds) == 1
    box, cls, probs = preds[0]
    assert 0 <= cls <= 6 and probs.shape == (7,) and np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert predict_frame(frame, model, "small", None, "cpu", detector=lambda g: np.empty((0, 4), int)) == []


def test_draw_predictions(frame):
    assert all(e.isascii() for e in EMOTIONS)
    draw_predictions(frame, [((10, 10, 40, 40), 3, np.full(7, 1 / 7))])
