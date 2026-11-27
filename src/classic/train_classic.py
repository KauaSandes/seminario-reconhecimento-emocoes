"""Treina um classificador clássico na FER-2013 e grava o JSON de métricas da validação.

O conjunto de teste nunca é carregado aqui: os hiperparâmetros são escolhidos na validação e o teste
fica para a avaliação final, uma vez por método (CLAUDE.md, seção 5).

Uso::

    python -m src.classic.train_classic                       # HOG + SVM linear, C escolhido na validação
    python -m src.classic.train_classic --features pixels     # linha de base: pixels brutos + SVM linear
    python -m src.classic.train_classic --model majority      # linha de base: classe majoritária
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.common import data
from src.common.metrics import evaluate, save_results
from src.common.utils import ensure_utf8_stdio, set_seed
from src.classic.features import FEATURES
from src.classic.preprocess import preprocess

DEFAULT_CS = (0.01, 0.1, 1.0)


def _load(split: str, args) -> tuple[np.ndarray, np.ndarray]:
    kw = {k: v for k, v in (("csv_path", args.csv_path), ("cache_path", args.cache_path),
                            ("splits_csv", args.splits_csv)) if v is not None}
    X, y, _ = data.load_fer2013(split, **kw)
    return X, y


def run_majority(args) -> dict:
    Xtr, ytr = _load("train", args)
    _, yva = _load("val", args)
    t0 = time.perf_counter()
    majority = int(np.bincount(ytr, minlength=data.NUM_CLASSES).argmax())
    train_time = time.perf_counter() - t0
    pred = np.full_like(yva, majority)
    print(f"classe majoritária do treino: {majority} ({data.EMOTIONS[majority]})")
    meta = {"method": "majority", "track": "baseline", "dataset": "fer2013", "split": "val",
            "degradation": "none", "level": 0, "train_time_s": train_time, "inference_ms_per_img": 0.0,
            "seed": args.seed}
    return evaluate(yva, pred, meta)


def run_linearsvc(args) -> dict:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC

    Xtr, ytr = _load("train", args)
    Xva, yva = _load("val", args)
    if args.max_train:
        Xtr, ytr = Xtr[: args.max_train], ytr[: args.max_train]
    featurize = FEATURES[args.features]
    prep = (lambda X: X) if args.no_preprocess else preprocess

    t0 = time.perf_counter()
    Ftr = featurize(prep(Xtr))
    feat_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    Fva = featurize(prep(Xva))
    feat_val_time = time.perf_counter() - t0
    print(f"características: {args.features}, {Ftr.shape[1]} dimensões, treino {len(Ftr)}, validação {len(Fva)} "
          f"({feat_time:.1f} s)")

    best = None
    for C in args.C:
        model = make_pipeline(StandardScaler(), LinearSVC(C=C, dual=False, max_iter=args.max_iter,
                                                          random_state=args.seed))
        t0 = time.perf_counter()
        model.fit(Ftr, ytr)
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        pred = model.predict(Fva)
        pred_time = time.perf_counter() - t0
        rec = evaluate(yva, pred, _meta(args, fit_time + feat_time,
                                        1000 * (feat_val_time + pred_time) / len(Fva)))
        print(f"  C={C:g}: acurácia {rec['accuracy']:.4f}, F1 macro {rec['f1_macro']:.4f} ({fit_time:.1f} s)")
        if best is None or rec["f1_macro"] > best[1]["f1_macro"]:
            best = (C, rec)
    print(f"melhor C na validação (pelo F1 macro): {best[0]:g}")
    return best[1]


def _meta(args, train_time: float, inference_ms: float) -> dict:
    return {"method": f"{args.features}_linearsvc", "track": "classic" if args.features != "pixels" else "baseline",
            "dataset": "fer2013", "split": "val", "degradation": "none", "level": 0,
            "train_time_s": train_time, "inference_ms_per_img": inference_ms, "seed": args.seed}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classificador clássico na FER-2013 (métricas na validação).")
    parser.add_argument("--model", choices=("linearsvc", "majority"), default="linearsvc")
    parser.add_argument("--features", choices=tuple(FEATURES), default="hog")
    parser.add_argument("--C", type=float, nargs="+", default=list(DEFAULT_CS),
                        help="valores de C testados; o melhor F1 macro na validação é o gravado")
    parser.add_argument("--no-preprocess", action="store_true", help="sem equalização e gaussiano")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--max-train", type=int, default=0, help="limita o treino (só para testes rápidos)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="aceito por convenção; os clássicos rodam em CPU")
    parser.add_argument("--results-dir", type=Path, default=None)
    # Caminhos alternativos dos dados (usados pelos testes).
    parser.add_argument("--csv-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cache-path", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--splits-csv", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    ensure_utf8_stdio()
    set_seed(args.seed)

    record = run_majority(args) if args.model == "majority" else run_linearsvc(args)
    path = save_results(record, args.results_dir)
    print(f"{record['method']}: acurácia {record['accuracy']:.4f}, F1 macro {record['f1_macro']:.4f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
