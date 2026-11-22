"""Laço de treino compartilhado por train_scratch e train_resnet."""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from src.cnn.datasets import make_loaders
from src.cnn.models import MODELS, save_checkpoint
from src.common.metrics import evaluate, save_results
from src.common.utils import set_seed


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, max_batches=None) -> float:
    """Uma época de treino; devolve a perda média por amostra."""
    model.train()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=scaler is not None):
            loss = criterion(model(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.inference_mode()
def predict(model, loader, device, criterion=None, amp=False, max_batches=None):
    """-> (y_true, y_pred, loss ou None, ms por imagem medido só no forward)."""
    model.eval()
    ys, ps, total, n, elapsed = [], [], 0.0, 0, 0.0
    cuda = torch.device(device).type == "cuda"
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            out = model(x)
        if cuda:
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t
        if criterion is not None:
            total += criterion(out.float(), y).item() * len(y)
        n += len(y)
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
    loss = total / n if criterion is not None and n else None
    return (np.concatenate(ys).astype(np.int64), np.concatenate(ps).astype(np.int64), loss,
            1000 * elapsed / max(n, 1))


def val_epoch(model, val_loader, criterion, device, amp, max_batches) -> tuple[float, float]:
    y, p, loss, _ = predict(model, val_loader, device, criterion, amp, max_batches)
    return loss, float((y == p).mean())


def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs, patience, scaler=None,
        log=print, max_batches=None, start_epoch=1) -> dict:
    """Treina com parada antecipada pela perda de validação e restaura o melhor estado."""
    history, best_loss, best_epoch, best_state, bad = [], float("inf"), start_epoch, None, 0
    for epoch in range(start_epoch, start_epoch + epochs):
        tl = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, max_batches)
        vl, va = val_epoch(model, val_loader, criterion, device, scaler is not None, max_batches)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_acc": va})
        log(f"epoca {epoch}: train_loss={tl:.4f} val_loss={vl:.4f} val_acc={va:.4f}")
        if vl < best_loss:
            best_loss, best_epoch, bad = vl, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                log(f"parada antecipada na epoca {epoch}")
                break
    if best_state is None:
        raise RuntimeError("treino divergiu: perda de validação não finita em todas as épocas; "
                           "reduza --lr ou use --no-amp")
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": best_loss, "history": history}


# ---- pedaços comuns dos scripts de treino ----

def make_parser(description: str, epochs: int, patience: int, lr: float) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--patience", type=int, default=patience)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--balance", choices=["none", "sampler", "loss"], default="none")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--limit-batches", type=int, default=None)
    return p


def setup(args, name: str, size: int | None = None):
    """Semente, loaders (só train/val), critério e scaler de AMP."""
    set_seed(args.seed)
    loaders = make_loaders(MODELS[name], args.batch_size, args.workers, args.seed, args.balance,
                           size=size, splits=("train", "val"))
    w = loaders.class_weights.to(args.device) if args.balance == "loss" else None
    scaler = torch.amp.GradScaler() if torch.device(args.device).type == "cuda" and not args.no_amp else None
    return loaders, torch.nn.CrossEntropyLoss(weight=w), scaler


def finish(args, model, name, size, loaders, scaler, best_epoch, best_val_loss, history, t0):
    """Grava checkpoint, avalia no val e grava o JSON de métricas."""
    train_time = time.perf_counter() - t0
    save_checkpoint(f"{args.ckpt_dir}/{name}.pt", model, name=name, size=size, epoch=best_epoch,
                    val_loss=best_val_loss, history=history, args=vars(args))
    y, p, _, ms = predict(model, loaders.val, args.device, amp=scaler is not None,
                          max_batches=args.limit_batches)
    meta = {"method": name, "track": "cnn", "dataset": "fer2013", "split": "val", "degradation": "none",
            "level": 0, "train_time_s": train_time, "inference_ms_per_img": ms, "seed": args.seed}
    rec = evaluate(y, p, meta)
    path = save_results(rec, out_dir=args.results_dir)
    print(f"acuracia={rec['accuracy']:.4f} f1_macro={rec['f1_macro']:.4f} json={path}")
    return rec
