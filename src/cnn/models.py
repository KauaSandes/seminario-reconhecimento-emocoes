"""Modelos da Frente A (CNN do zero e ResNet-18) e checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

#: nome do método -> modo de src.cnn.datasets
MODELS = {"cnn_scratch": "small", "resnet18": "resnet"}


def _block(cin: int, cout: int) -> nn.Sequential:
    layers = []
    for c in (cin, cout):
        layers += [nn.Conv2d(c, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


class SmallCNN(nn.Module):
    """3 blocos (2x conv 3x3 + BN + ReLU, MaxPool): 1x48x48 -> 256x6x6 -> classes."""

    def __init__(self, num_classes: int = 7, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(_block(1, 64), _block(64, 128), _block(128, 256))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(256 * 6 * 6, 256), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def build_resnet18(num_classes: int = 7, pretrained: bool = True) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(512, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Liga/desliga o gradiente de tudo menos ``model.fc`` (sempre treinável)."""
    for name, p in model.named_parameters():
        p.requires_grad = trainable or name.startswith("fc.")


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"modelo deve ser um de {list(MODELS)}, não {name!r}")
    return SmallCNN() if name == "cnn_scratch" else build_resnet18(pretrained=pretrained)


def save_checkpoint(path, model: nn.Module, *, name: str, size: int | None, epoch: int,
                    val_loss: float, history: list[dict], args: dict) -> None:
    # args vira str/int/float/bool/None para permitir torch.load(weights_only=True)
    simple = {k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for k, v in args.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"name": name, "mode": MODELS[name], "size": size, "state_dict": model.state_dict(),
                "epoch": int(epoch), "val_loss": float(val_loss), "history": history, "args": simple}, path)


def load_checkpoint(path, device="cpu"):
    """Reconstrói o modelo do checkpoint (sem baixar pesos) e devolve ``(model, ckpt)`` em eval."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
