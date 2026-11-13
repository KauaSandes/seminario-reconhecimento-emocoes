"""Prepara os recursos do PPTX do checkpoint 2 em ``pptx_build/`` (fora do git).

- fundos do tema UFG (lib/layouts/*.pdf) renderizados sobre o UFGBlue em 1920x1080;
- logos em PNG (o ufg_black.pdf vira PNG);
- ``chart_data.json``: só números agregados para os gráficos nativos do PowerPoint
  (remoções por motivo, distribuição de classes, histograma do desvio padrão).

Uso (da raiz do repositório)::

    python reports/slides/checkpoint2-dados/pptx_assets.py
    node reports/slides/checkpoint2-dados/build_pptx.js
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from src.common.clean import DEFAULT_STD_THRESHOLD  # noqa: E402
from src.common.data import EMOTIONS, fer2013_arrays, read_clean_csv  # noqa: E402

LIB = HERE / "lib"
OUT = HERE / "pptx_build"
UFGBLUE = (0, 114 / 255, 185 / 255)


def render_backgrounds() -> None:
    for name in ("intro_bg", "vertical_bg", "horizontal_bg"):
        src = pymupdf.open(LIB / "layouts" / f"{name}.pdf")
        doc = pymupdf.open()
        page = doc.new_page(width=src[0].rect.width, height=src[0].rect.height)
        page.draw_rect(page.rect, color=None, fill=UFGBLUE)
        page.show_pdf_page(page.rect, src, 0)
        zoom = 1920 / page.rect.width
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(OUT / f"{name}.png")


def render_logos() -> None:
    ufg = pymupdf.open(LIB / "logos" / "ufg_black.pdf")[0]
    ufg.get_pixmap(matrix=pymupdf.Matrix(6, 6), alpha=True).save(OUT / "ufg_black.png")
    for name in ("INF-compact-white.png", "INF-full-white.png"):
        shutil.copy(LIB / "logos" / name, OUT / name)
    shutil.copy(REPO / "reports" / "figures" / "degradations_example_face.png", OUT / "degradations_example_face.png")


def chart_data() -> dict:
    decision = read_clean_csv()
    train = decision[decision["split"] == "train"]
    removed = train[~train["keep"]]
    reasons = removed["reason"].str.split(";").explode().value_counts()
    removal_labels = {
        "low_std": "Imagem vazia (desvio < 5)",
        "label_conflict": "Cópias com rótulos diferentes",
        "dup_cross_split_leak": "Cópia de imagem da validação/teste",
        "dup_within_split": "Duplicata dentro do treino",
    }
    before = train["label"].value_counts(normalize=True).reindex(range(7), fill_value=0) * 100
    after = train.loc[train["keep"], "label"].value_counts(normalize=True).reindex(range(7), fill_value=0) * 100

    X = fer2013_arrays()["X"]
    std = X.reshape(len(X), -1).astype(np.float64).std(axis=1)
    edges = np.arange(0, np.ceil(std.max()) + 2, 2.0)       # caixas de 2 em 2
    counts, _ = np.histogram(std, bins=edges)
    low = edges[:-1] < DEFAULT_STD_THRESHOLD
    return {
        "removals": {"labels": list(removal_labels.values()),
                     "values": [int(reasons.get(k, 0)) for k in removal_labels],
                     "total": int(len(removed))},
        "distribution": {"labels": list(EMOTIONS),
                         "before": [round(float(v), 2) for v in before],
                         "after": [round(float(v), 2) for v in after],
                         "n_before": int(len(train)), "n_after": int(train["keep"].sum())},
        "histogram": {"labels": [f"{int(e)}" for e in edges[:-1]],
                      "low": [int(c) if (lo and c) else None for c, lo in zip(counts, low)],
                      "high": [int(c) if (not lo and c) else None for c, lo in zip(counts, low)],
                      "n_low": int((std < DEFAULT_STD_THRESHOLD).sum()),
                      "threshold": DEFAULT_STD_THRESHOLD},
    }


def main() -> int:
    OUT.mkdir(exist_ok=True)
    render_backgrounds()
    render_logos()
    data = chart_data()
    (OUT / "chart_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("recursos em", OUT, "| remoções:", data["removals"]["values"], "| vazias:", data["histogram"]["n_low"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
