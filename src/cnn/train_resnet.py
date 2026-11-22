"""Transfer learning da ResNet-18 na FER-2013 (train/val): ``python -m src.cnn.train_resnet``.

Fase 1: só a camada final. Fase 2: rede inteira com taxa menor e parada antecipada.
"""
from __future__ import annotations

import time

import torch

from src.cnn import engine
from src.cnn.models import build_model, set_backbone_trainable

NAME = "resnet18"


def main(argv=None):
    p = engine.make_parser(__doc__, epochs=30, patience=5, lr=1e-4)
    p.add_argument("--size", type=int, choices=[224, 112], default=224)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--head-epochs", type=int, default=3)
    p.add_argument("--no-pretrained", action="store_true", help="só para testes")
    args = p.parse_args(argv)
    t0 = time.perf_counter()
    loaders, criterion, scaler = engine.setup(args, NAME, args.size)
    model = build_model(NAME, pretrained=not args.no_pretrained).to(args.device)
    mb, amp = args.limit_batches, scaler is not None

    set_backbone_trainable(model, False)
    opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr_head)
    history = []
    for epoch in range(1, args.head_epochs + 1):
        tl = engine.train_one_epoch(model, loaders.train, criterion, opt, args.device, scaler, mb)
        vl, va = engine.val_epoch(model, loaders.val, criterion, args.device, amp, mb)
        history.append({"epoch": epoch, "train_loss": tl, "val_loss": vl, "val_acc": va})
        print(f"epoca {epoch} (cabeca): train_loss={tl:.4f} val_loss={vl:.4f} val_acc={va:.4f}")

    set_backbone_trainable(model, True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    r = engine.fit(model, loaders.train, loaders.val, criterion, opt, args.device, args.epochs,
                   args.patience, scaler, max_batches=mb, start_epoch=args.head_epochs + 1)
    return engine.finish(args, model, NAME, args.size, loaders, scaler, r["best_epoch"],
                         r["best_val_loss"], history + r["history"], t0)


if __name__ == "__main__":
    main()
