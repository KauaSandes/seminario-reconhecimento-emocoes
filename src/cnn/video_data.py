"""Sequências do CK+ para o modelo de vídeo (CNN por quadro + LSTM).

As sequências vêm de ``src.common.data.load_ckplus(..., mode="sequence")``, já com o
rosto recortado, sem "desprezo" e com rótulos na ordem da FER-2013.

Suposição sobre a API da Frente B (ajustar em ``_load_sequences`` se mudar):
    load_ckplus(fold, part, mode="sequence") -> (sequences, labels)
        sequences: lista de arrays uint8 [n_quadros, H, W], do neutro ao pico
        labels: int [N] em 0..6
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import functional as F

from src.cnn.datasets import MODES, build_transform
from src.common import data as common_data

PARTS = ("train", "test")


def _load_sequences(fold: int, part: str):
    sequences, labels = common_data.load_ckplus(fold, part=part, mode="sequence")
    return list(sequences), np.asarray(labels)


def sample_frame_indices(n_frames: int, T: int) -> np.ndarray:
    """T índices uniformes do neutro (0) ao pico (n-1); repete o último se n < T."""
    if n_frames <= 0:
        raise ValueError("sequência vazia")
    if n_frames >= T:
        return np.round(np.linspace(0, n_frames - 1, T)).astype(int)
    return np.concatenate([np.arange(n_frames), np.full(T - n_frames, n_frames - 1)])


class CKSequenceDataset(Dataset):
    """Devolve ``(x [T, C, H, W], rótulo)`` para cada sequência de um fold do CK+.

    Args:
        fold: índice do fold (GroupKFold por pessoa, definido pela Frente B).
        part: "train" ou "test" dentro do fold.
        T: número de quadros amostrados.
        mode, size: mesma transformação da FER-2013 ("small" -> 1x48x48;
            "resnet" -> 3xSxS com normalização ImageNet). Quadros de outro tamanho são
            redimensionados para 48x48 antes.
    """

    def __init__(self, fold: int, part: str = "train", T: int = 16, mode: str = "small",
                 size: int | None = None, sequences=None, labels=None):
        if part not in PARTS:
            raise ValueError(f"part deve ser um de {PARTS}")
        if mode not in MODES:
            raise ValueError(f"mode deve ser um de {MODES}")
        if sequences is None:
            sequences, labels = _load_sequences(fold, part)
        if len(sequences) != len(labels):
            raise ValueError("sequences e labels com tamanhos diferentes")
        self.fold, self.part, self.T = fold, part, T
        self.sequences = sequences
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
        self.transform = build_transform(mode, augment=False, size=size)

    def __len__(self) -> int:
        return len(self.labels)

    def _frame(self, img: np.ndarray) -> torch.Tensor:
        if img.shape != (48, 48):  # mesmo ponto de partida da FER-2013: uint8 48x48
            t = F.resize(torch.from_numpy(np.ascontiguousarray(img))[None], [48, 48], antialias=True)
            img = t[0].numpy()
        return self.transform(img)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = np.asarray(self.sequences[idx])
        frames = [self._frame(seq[i]) for i in sample_frame_indices(len(seq), self.T)]
        return torch.stack(frames), self.labels[idx]


@torch.no_grad()
def extract_features(backbone: torch.nn.Module, fold: int, model_name: str, T: int = 16,
                     mode: str = "small", size: int | None = None, batch_size: int = 8,
                     device: str | torch.device = "cpu", out_dir: str | Path = "data/processed") -> Path:
    """Passa cada quadro por uma CNN congelada e salva os vetores do fold.

    ``backbone`` deve devolver um vetor por imagem ([B, D]); para a ResNet-18, use
    ``model.fc = torch.nn.Identity()``. Arquivo salvo:
    ``<out_dir>/ck_features_<model_name>_fold<fold>.pt`` com
    ``{"train": {"features": [N, T, D], "labels": [N]}, "test": {...}, "model", "T", "fold"}``.
    """
    device = torch.device(device)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    out: dict = {"model": model_name, "T": T, "fold": fold, "mode": mode}
    for part in PARTS:
        ds = CKSequenceDataset(fold, part, T=T, mode=mode, size=size)
        feats, labels = [], []
        for x, y in DataLoader(ds, batch_size=batch_size, shuffle=False):
            b = x.shape[0]
            f = backbone(x.flatten(0, 1).to(device))  # [B*T, D]
            feats.append(f.reshape(b, T, -1).float().cpu())
            labels.append(y)
        out[part] = {"features": torch.cat(feats), "labels": torch.cat(labels)}

    path = Path(out_dir) / f"ck_features_{model_name}_fold{fold}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return path
