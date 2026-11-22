"""Teste de fumaça do pipeline de dados da Frente A.

1. Uma CNN minúscula precisa decorar um lote fixo de 64 imagens de TREINO (acurácia
   perto de 100%) em poucas dezenas de passos. Se não decorar, há erro em dados,
   rótulos ou transformações.
2. ``--bench-loader``: mede a vazão do DataLoader de treino (imagens/s) para vários
   ``num_workers``.
3. Grava um JSON de exemplo com ``src.common.metrics.save_results`` (method="smoke").

Não usa o conjunto de teste. Não é treino de verdade.

Uso:
    python scripts/smoke_test.py --mode small
    python scripts/smoke_test.py --mode resnet --size 224 --bench-loader --workers 0 2 4 8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from src.cnn.datasets import FERDataset, make_loaders  # noqa: E402
from src.common import metrics as common_metrics  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TinyCNN(nn.Module):
    """CNN mínima que serve para os dois modos (1x48x48 ou 3xSxS)."""

    def __init__(self, in_channels: int, num_classes: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(6), nn.Flatten(), nn.Linear(32 * 36, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def fixed_batch(mode: str, n: int, size: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """As n primeiras imagens do treino, pelo mesmo FERDataset usado no treino (sem aumento)."""
    ds = FERDataset("train", mode=mode, augment=False, size=size)
    xs, ys = zip(*(ds[i] for i in range(n)))
    return torch.stack(xs), torch.stack(ys)


def run_smoke(mode: str = "small", n: int = 64, steps: int = 60, lr: float = 3e-3, size: int | None = None,
              device: str = "cpu", seed: int = 42) -> dict:
    set_seed(seed)
    dev = torch.device(device)
    x, y = fixed_batch(mode, n, size)
    x, y = x.to(dev), y.to(dev)
    model = TinyCNN(in_channels=x.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    amp = dev.type == "cuda"
    dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16

    t0 = time.perf_counter()
    losses = []
    model.train()
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=dtype, enabled=amp):
            loss = nn.functional.cross_entropy(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_time = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        pred = model(x).argmax(1)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.perf_counter() - t1) * 1e3 / n
    acc = (pred == y).float().mean().item()
    return {"mode": mode, "input_shape": list(x.shape), "accuracy": acc, "first_loss": losses[0],
            "last_loss": losses[-1], "train_time_s": train_time, "inference_ms_per_img": infer_ms,
            "y_true": y.cpu().numpy(), "y_pred": pred.cpu().numpy(), "seed": seed}


def save_smoke_json(result: dict, out_dir: str | Path | None = None) -> Path:
    """Valida a integração com o contrato gravando o JSON pelo src.common.metrics.

    ``split="train"`` porque o lote fixo vem do treino (o teste não é tocado).
    """
    meta = {"method": "smoke", "track": "cnn", "dataset": "fer2013", "split": "train",
            "degradation": "none", "level": 0, "train_time_s": result["train_time_s"],
            "inference_ms_per_img": result["inference_ms_per_img"], "seed": result["seed"]}
    record = common_metrics.evaluate(result["y_true"], result["y_pred"], meta)
    return common_metrics.save_results(record, out_dir)


def bench_loader(mode: str, workers: list[int], batch_size: int, n_batches: int, size: int | None,
                 seed: int) -> list[tuple[int, float]]:
    rows = []
    for nw in workers:
        loaders = make_loaders(mode, batch_size=batch_size, num_workers=nw, seed=seed, size=size, splits=("train",))
        it = iter(loaders.train)
        next(it)  # aquece (sobe os workers)
        t0 = time.perf_counter()
        seen = 0
        for _, (xb, _) in zip(range(n_batches), it):
            seen += xb.shape[0]
        rows.append((nw, seen / (time.perf_counter() - t0)))
        del it, loaders
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["small", "resnet"], default="small")
    p.add_argument("--size", type=int, default=None, help="lado no modo resnet (224 ou 112)")
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--min-acc", type=float, default=0.95)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bench-loader", action="store_true")
    p.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--bench-batches", type=int, default=50)
    p.add_argument("--no-json", action="store_true", help="não gravar o JSON de exemplo")
    p.add_argument("--out-dir", default=None, help="pasta do JSON (padrão: results/)")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    r = run_smoke(args.mode, args.n, args.steps, size=args.size, device=args.device, seed=args.seed)
    ok = r["accuracy"] >= args.min_acc
    print(f"[{args.mode}] entrada {r['input_shape']}: perda {r['first_loss']:.3f} -> {r['last_loss']:.4f}, "
          f"acurácia no lote fixo {r['accuracy']:.1%} em {args.steps} passos ({r['train_time_s']:.1f} s) "
          f"-> {'OK' if ok else 'FALHOU'}")

    if args.bench_loader:
        print(f"\nVazão do DataLoader de treino (mode={args.mode}, lote {args.batch_size}, com aumento):")
        print("| num_workers | imagens/s |\n|---|---|")
        for nw, ips in bench_loader(args.mode, args.workers, args.batch_size, args.bench_batches, args.size,
                                    args.seed):
            print(f"| {nw} | {ips:.0f} |")

    if not args.no_json:
        path = save_smoke_json(r, args.out_dir)
        print(f"\nJSON de exemplo: {path}")
        if path and Path(path).is_file():
            print(json.dumps({k: v for k, v in json.loads(Path(path).read_text()).items()
                              if k not in ("confusion_matrix", "f1_per_class")}, ensure_ascii=False))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
