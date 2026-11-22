"""Treina a CNN do zero na FER-2013 (train/val): ``python -m src.cnn.train_scratch``."""
from __future__ import annotations

import time

import torch

from src.cnn import engine
from src.cnn.models import build_model

NAME = "cnn_scratch"


def main(argv=None):
    args = engine.make_parser(__doc__, epochs=60, patience=8, lr=1e-3).parse_args(argv)
    t0 = time.perf_counter()
    loaders, criterion, scaler = engine.setup(args, NAME)
    model = build_model(NAME).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    r = engine.fit(model, loaders.train, loaders.val, criterion, opt, args.device, args.epochs,
                   args.patience, scaler, max_batches=args.limit_batches)
    return engine.finish(args, model, NAME, None, loaders, scaler, r["best_epoch"],
                         r["best_val_loss"], r["history"], t0)


if __name__ == "__main__":
    main()
