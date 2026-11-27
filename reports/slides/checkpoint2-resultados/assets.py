"""Prepara os recursos do deck do checkpoint 2 (resultados preliminares) em ``build/`` (fora do git).

- fundos e logos do tema UFG (reaproveitados de ../checkpoint2-dados/lib);
- ``deck_data.json``: números lidos de ``results/*.json`` e de ``splits/fer2013_clean.csv``.

Uso (da raiz do repositório)::

    python reports/slides/checkpoint2-resultados/assets.py
    node reports/slides/checkpoint2-resultados/build_pptx.js
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.common.data import EMOTIONS, read_clean_csv  # noqa: E402

LIB = HERE.parent / "checkpoint2-dados" / "lib"
OUT = HERE / "build"
UFGBLUE = (0, 114 / 255, 185 / 255)
# Ordem de exibição: linhas de base, clássico, redes.
METHODS = [("majority", "Classe majoritária"), ("pixels_linearsvc", "Pixels + SVM"),
           ("hog_linearsvc", "HOG + SVM"), ("cnn_scratch", "CNN do zero"), ("resnet18", "ResNet-18")]


def render_backgrounds() -> None:
    for name in ("intro_bg", "vertical_bg"):
        src = pymupdf.open(LIB / "layouts" / f"{name}.pdf")
        doc = pymupdf.open()
        page = doc.new_page(width=src[0].rect.width, height=src[0].rect.height)
        page.draw_rect(page.rect, color=None, fill=UFGBLUE)
        page.show_pdf_page(page.rect, src, 0)
        zoom = 1920 / page.rect.width
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(OUT / f"{name}.png")
    ufg = pymupdf.open(LIB / "logos" / "ufg_black.pdf")[0]
    ufg.get_pixmap(matrix=pymupdf.Matrix(6, 6), alpha=True).save(OUT / "ufg_black.png")
    for name in ("INF-compact-white.png", "INF-full-white.png"):
        shutil.copy(LIB / "logos" / name, OUT / name)


def deck_data() -> dict:
    results = {}
    for key, label in METHODS:
        path = REPO / "results" / f"{key}_fer2013_val_none_0.json"
        if path.is_file():
            r = json.loads(path.read_text(encoding="utf-8"))
            results[key] = {"label": label, "accuracy": r["accuracy"], "f1_macro": r["f1_macro"],
                            "f1_per_class": r["f1_per_class"], "confusion_matrix": r["confusion_matrix"],
                            "train_time_s": r["train_time_s"], "inference_ms_per_img": r["inference_ms_per_img"],
                            "track": r["track"]}
    d = read_clean_csv()
    train = d[d["split"] == "train"]
    removed = train[~train["keep"]]
    kept = train[train["keep"]]
    return {
        "emotions": list(EMOTIONS),
        "results": results,
        "order": [k for k, _ in METHODS if k in results],
        "data": {"train_before": int(len(train)), "train_after": int(len(kept)), "removed": int(len(removed)),
                 "leak": int(removed["reason"].str.contains("dup_cross_split_leak").sum()),
                 "val": int((d["split"] == "val").sum()), "test": int((d["split"] == "test").sum()),
                 "disgust_share": float((kept["label"] == 1).mean())},
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    render_backgrounds()
    data = deck_data()
    (OUT / "deck_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("recursos em", OUT)
    for k in data["order"]:
        r = data["results"][k]
        print(f"  {r['label']:20s} acurácia {r['accuracy']:.4f}  F1 macro {r['f1_macro']:.4f}")
    print("  dados:", data["data"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
