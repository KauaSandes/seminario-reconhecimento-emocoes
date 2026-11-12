"""Datasets e DataLoaders da FER-2013 para as redes da Frente A.

Os dados vêm só de ``src.common`` (contrato, CLAUDE.md seção 5). Este módulo apenas
converte os arrays para tensores, aplica aumento de dados e monta os loaders.

Ordem das operações em cada imagem (uint8, 48x48):

    degradação (src.common.degrade, só teste) -> [0, 1] -> aumento (só treino)
    -> mode="small": 1x48x48
    -> mode="resnet": redimensiona para 224 (ou 112), repete o cinza em 3 canais,
       normalização do ImageNet

O aumento roda em 48x48, antes do redimensionamento, para o padding de 4 px e a
rotação valerem na resolução original (e ficar mais barato).

Suposições sobre a API da Frente B (ajustar aqui se a assinatura publicada mudar):
    load_fer2013(split) -> (images uint8 [N, 48, 48], labels int [N]),
        split em {"train", "val", "test"}
    class_weights(labels) -> array [7] com o peso de cada classe
    add_gaussian_noise(img, sigma), gaussian_blur(img, ksize), downsample(img, size),
        cada uma recebendo e devolvendo uma imagem uint8 2D
"""
from __future__ import annotations

import random
from typing import Callable, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import v2

from src.common import clean as common_clean
from src.common import data as common_data
from src.common import degrade as common_degrade

NUM_CLASSES = 7
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODES = ("small", "resnet")
SPLITS = ("train", "val", "test")

# Nome da degradação no JSON de resultados -> função de src.common.degrade.
DEGRADATIONS: dict[str, str] = {
    "noise": "add_gaussian_noise",   # nível = sigma (5, 10, 20)
    "blur": "gaussian_blur",         # nível = ksize (3, 5, 7)
    "resolution": "downsample",      # nível = tamanho intermediário (24, 12)
}


class RepeatChannels(torch.nn.Module):
    """1xHxW -> CxHxW repetindo o canal (classe, não lambda, para funcionar com workers no Windows)."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.expand(self.channels, -1, -1).clone()


def build_transform(mode: str, augment: bool, size: int | None = None) -> Callable[[np.ndarray], torch.Tensor]:
    """Transformação de uma imagem uint8 2D para o tensor de entrada da rede."""
    if mode not in MODES:
        raise ValueError(f"mode deve ser um de {MODES}, não {mode!r}")
    steps: list = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]  # HxW uint8 -> 1xHxW em [0, 1]
    if augment:
        steps += [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10),
            v2.RandomCrop(48, padding=4),
        ]
    if mode == "resnet":
        size = size or 224
        steps += [
            v2.Resize((size, size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            RepeatChannels(3),  # repete o canal cinza
            v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    elif size not in (None, 48):
        steps.append(v2.Resize((size, size), antialias=True))
    return v2.Compose(steps)


def apply_degradation(images: np.ndarray, degradation: str | None, level: float | int | None,
                      seed: int = 42) -> np.ndarray:
    """Aplica uma degradação de ``src.common.degrade`` a todas as imagens, uma vez.

    Feito antes das transformações e fora do ``__getitem__`` para o ruído ficar fixo
    entre épocas e igual ao da Frente B com a mesma semente.
    """
    if degradation in (None, "none"):
        return images
    if degradation not in DEGRADATIONS:
        raise ValueError(f"degradação desconhecida {degradation!r}; use uma de {list(DEGRADATIONS)}")
    fn = getattr(common_degrade, DEGRADATIONS[degradation])
    random.seed(seed)
    np.random.seed(seed)
    out = np.stack([np.asarray(fn(img, level)) for img in images])
    return out.astype(np.uint8, copy=False)


class FERDataset(Dataset):
    """FER-2013 a partir dos arrays de ``src.common.data.load_fer2013``.

    Args:
        split: "train", "val" ou "test".
        mode: "small" (1x48x48 em [0, 1]) ou "resnet" (3xSxS, normalização ImageNet).
        augment: espelhamento, rotação de ±10° e recorte com padding de 4 px. Só no treino.
        size: lado da entrada no modo "resnet" (224 padrão, 112 como opção).
        degradation, level: degradação de teste (ver ``DEGRADATIONS``); "none" ou None desliga.
        images, labels: arrays já carregados (evita carregar de novo); por padrão chama
            ``load_fer2013(split)``.
    """

    def __init__(self, split: str, mode: str = "small", augment: bool = False, size: int | None = None,
                 degradation: str | None = None, level: float | int | None = None, seed: int = 42,
                 images: np.ndarray | None = None, labels: np.ndarray | None = None):
        if split not in SPLITS:
            raise ValueError(f"split deve ser um de {SPLITS}, não {split!r}")
        if augment and split != "train":
            raise ValueError("aumento de dados só no treino")
        if degradation not in (None, "none") and split != "test":
            raise ValueError("degradações só no teste (contrato, seção 5)")
        if images is None:
            images, labels = common_data.load_fer2013(split)
        images = np.asarray(images)
        labels = np.asarray(labels)
        if images.ndim != 3 or images.dtype != np.uint8:
            raise ValueError(f"esperado uint8 [N, H, W], recebido {images.dtype} {images.shape}")
        if len(images) != len(labels):
            raise ValueError("images e labels com tamanhos diferentes")
        if labels.size and (labels.min() < 0 or labels.max() >= NUM_CLASSES):
            raise ValueError("rótulos fora de 0..6")

        self.split, self.mode, self.augment = split, mode, augment
        self.images = apply_degradation(images, degradation, level, seed)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.transform = build_transform(mode, augment, size)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[idx]), self.labels[idx]


def seed_worker(worker_id: int) -> None:
    """Semente de cada worker derivada da semente do loader (receita da doc do PyTorch)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def class_weight_tensor(labels: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Pesos de classe de ``src.common.clean.class_weights`` como tensor float [7]."""
    w = torch.as_tensor(np.asarray(common_clean.class_weights(np.asarray(labels))), dtype=torch.float32)
    if w.shape != (NUM_CLASSES,):
        raise ValueError(f"class_weights deveria ter {NUM_CLASSES} valores, tem {tuple(w.shape)}")
    return w


class Loaders(NamedTuple):
    train: DataLoader | None
    val: DataLoader | None
    test: DataLoader | None
    class_weights: torch.Tensor | None  # para CrossEntropyLoss(weight=...) quando balance="loss"


def make_loaders(mode: str = "small", batch_size: int = 64, num_workers: int = 4, seed: int = 42,
                 balance: str = "none", size: int | None = None, augment: bool = True,
                 degradation: str | None = None, level: float | int | None = None,
                 splits: tuple[str, ...] = SPLITS, pin_memory: bool | None = None) -> Loaders:
    """Monta os DataLoaders de treino, validação e teste.

    Args:
        balance: "none"; "sampler" (WeightedRandomSampler no treino); ou "loss"
            (devolve ``class_weights`` para a perda). Os dois usam
            ``src.common.clean.class_weights`` e não devem ser combinados.
        augment: aumento de dados no treino (val e teste nunca têm).
        degradation, level: aplicados só ao teste.
        splits: quais loaders criar; os outros voltam como None (ex.: não carregar o teste
            durante o desenvolvimento).
    """
    if balance not in ("none", "sampler", "loss"):
        raise ValueError("balance deve ser 'none', 'sampler' ou 'loss'")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                  worker_init_fn=seed_worker, persistent_workers=num_workers > 0)
    out: dict[str, DataLoader | None] = {s: None for s in SPLITS}
    weights = None

    if "train" in splits:
        ds = FERDataset("train", mode, augment=augment, size=size, seed=seed)
        g = torch.Generator().manual_seed(seed)
        if balance != "none":
            weights = class_weight_tensor(ds.labels)
        if balance == "sampler":
            sampler = WeightedRandomSampler(weights[ds.labels].double(), num_samples=len(ds),
                                            replacement=True, generator=g)
            out["train"] = DataLoader(ds, sampler=sampler, generator=g, drop_last=True, **common)
        else:
            out["train"] = DataLoader(ds, shuffle=True, generator=g, drop_last=True, **common)
    for split in ("val", "test"):
        if split in splits:
            deg = (degradation, level) if split == "test" else (None, None)
            ds = FERDataset(split, mode, augment=False, size=size, degradation=deg[0], level=deg[1], seed=seed)
            out[split] = DataLoader(ds, shuffle=False, **common)

    return Loaders(out["train"], out["val"], out["test"], weights if balance == "loss" else None)
