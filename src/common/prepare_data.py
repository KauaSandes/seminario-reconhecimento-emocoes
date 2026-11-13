"""Prepara tudo o que as duas frentes usam, depois do download. Um único comando::

    python -m src.common.prepare_data [--std-threshold 5] [--skip-face-check] [--force]

Etapas:

1. FER-2013: confere ``data/raw/fer2013/fer2013.csv`` (estrutura e contagens oficiais) e
   monta o cache ``data/processed/fer2013.npz``.
2. Limpeza: gera ``splits/fer2013_clean.csv`` (versionado), ``reports/data_cleaning.md``
   e, para conferência visual, a grade e o histograma do desvio padrão em
   ``data/processed/inspection/`` (fora do git).
3. CK+: se os dados estiverem em ``data/raw/ckplus/``, recorta os rostos, grava
   ``data/processed/ckplus.npz`` e gera ``splits/ckplus_folds.csv`` (GroupKFold(10) por
   pessoa). Se não estiverem, avisa e segue.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from src.common import clean, data
from src.common.utils import ensure_utf8_stdio, sha256_file


def prepare_fer2013(std_threshold: float, face_check: bool, force: bool) -> bool:
    if not data.FER2013_CSV.is_file():
        print(f"[FER-2013] {data.FER2013_CSV} não existe. Rode antes: python scripts/download_data.py",
              file=sys.stderr)
        return False
    t0 = time.perf_counter()
    try:
        arr = data.fer2013_arrays(force=force)
    except ValueError as exc:
        print(f"[FER-2013] CSV inválido:\n{exc}", file=sys.stderr)
        return False
    counts = {u: int((arr["split"] == data.USAGE_TO_SPLIT[u]).sum()) for u in data.FER2013_SIZES}
    if counts != data.FER2013_SIZES:
        print(f"[FER-2013] contagens {counts} diferentes das oficiais {data.FER2013_SIZES}. "
              f"Rode python scripts/download_data.py --force.", file=sys.stderr)
        return False
    print(f"[FER-2013] {len(arr['y'])} imagens carregadas ({time.perf_counter() - t0:.1f}s), cache em "
          f"{data.FER2013_CACHE}")

    digest = sha256_file(data.FER2013_CSV)
    print(f"[FER-2013] SHA-256 do CSV: {digest}")
    print(f"[limpeza] limiar de std = {std_threshold:g}; Viola-Jones: {'sim' if face_check else 'não'}")
    t0 = time.perf_counter()
    df = clean.clean_fer2013(arr["X"], arr["y"], arr["split"], arr["ids"],
                             std_threshold=std_threshold, face_check=face_check)
    csv_path = clean.write_clean_csv(df)
    grid = clean.save_low_std_grid(arr["X"], df, std_threshold=std_threshold)
    report = clean.write_report(df, std_threshold=std_threshold, csv_sha256=digest, face_check=face_check,
                                grid_path=grid)
    print(f"[limpeza] concluída em {time.perf_counter() - t0:.1f}s")
    for split in data.SPLITS:
        part = df[df["split"] == split]
        print(f"[limpeza]   {split:5s}: {len(part):6d} -> {int(part['keep'].sum()):6d} "
              f"(removidas {int((~part['keep']).sum())})")
    weights = clean.class_weights(df.loc[(df["split"] == "train") & df["keep"], "label"].to_numpy())
    print("[limpeza]   pesos de classe (treino): "
          + ", ".join(f"{name}={w:.2f}" for name, w in zip(data.EMOTIONS, weights)))
    print(f"[limpeza] decisão: {csv_path}\n[limpeza] relatório: {report}\n[limpeza] grade para conferir o "
          f"limiar: {grid}")
    return True


def prepare_ckplus(force: bool) -> bool:
    if data.find_ckplus_dirs() is None:
        print(f"[CK+] ainda não disponível em {data.CKPLUS_RAW_DIR}; seguindo sem o CK+ "
              f"(veja data/raw/ckplus/LEIA-ME.txt).")
        return False
    t0 = time.perf_counter()
    cache = data.build_ckplus_cache(force=force)
    folds = data.make_ckplus_folds(cache)
    with np.load(cache, allow_pickle=False) as c:
        n_seq, n_frames = len(c["labels"]), len(c["frames"])
        n_subj = len(np.unique(c["subjects"]))
        detected = float(c["face_detected"].mean())
        per_class = np.bincount(c["labels"], minlength=data.NUM_CLASSES)
        skipped = dict(zip(c["skipped_keys"].tolist(), c["skipped"].tolist()))
    print(f"[CK+] {n_seq} sequências de {n_subj} pessoas ({n_frames} quadros) em {time.perf_counter() - t0:.1f}s")
    print(f"[CK+]   rosto detectado no 1º quadro em {100 * detected:.1f}% das sequências (resto: recorte central)")
    print("[CK+]   sequências por classe: " + ", ".join(f"{n}={k}" for n, k in zip(data.EMOTIONS, per_class)))
    print(f"[CK+]   descartadas: {skipped}")
    print(f"[CK+] cache: {cache}\n[CK+] folds: {folds}")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepara a FER-2013 (cache e limpeza) e o CK+ (recorte e folds).")
    parser.add_argument("--std-threshold", type=float, default=clean.DEFAULT_STD_THRESHOLD,
                        help="desvio padrão abaixo do qual a imagem é 'quase constante' (padrão: 5)")
    parser.add_argument("--skip-face-check", action="store_true",
                        help="não rodar o Viola-Jones de sinalização na FER-2013 (mais rápido)")
    parser.add_argument("--force", action="store_true", help="refazer os caches mesmo se existirem")
    args = parser.parse_args(argv)
    ensure_utf8_stdio()

    ok = prepare_fer2013(args.std_threshold, not args.skip_face_check, args.force)
    prepare_ckplus(args.force)
    if not ok:
        return 1
    print("\nPronto. Use src.common.data.load_fer2013(split) e load_ckplus(fold, part, mode).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
