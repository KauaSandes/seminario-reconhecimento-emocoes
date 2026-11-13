"""Pré-baixa os pesos do ImageNet da ResNet-18 para o cache local do torch.

Assim o treino da semana 2 não depende de internet. O cache fica em
``torch.hub.get_dir()`` (por padrão ``~/.cache/torch/hub``); para mudar, defina
``TORCH_HOME`` (no Colab, aponte para o Drive para não baixar a cada sessão).

Uso:
    python scripts/cache_pretrained.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torchvision
from torchvision.models import ResNet18_Weights


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    weights = ResNet18_Weights.IMAGENET1K_V1
    torchvision.models.resnet18(weights=weights)  # baixa se ainda não estiver no cache

    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not ckpt.is_file():
        sys.exit(f"Pesos não encontrados em {ckpt}")

    # Confere que carrega a partir do arquivo local, sem rede.
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    model = torchvision.models.resnet18()
    model.load_state_dict(state)
    print(f"TORCH_HOME = {os.environ.get('TORCH_HOME', '(padrão)')}")
    print(f"Pesos em cache: {ckpt} ({ckpt.stat().st_size / 2**20:.1f} MiB)")
    print("OK: resnet18(weights='IMAGENET1K_V1') vai carregar sem internet.")


if __name__ == "__main__":
    main()
