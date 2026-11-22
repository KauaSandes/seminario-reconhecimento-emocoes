"""Figuras dos slides do checkpoint 2 (preparação e limpeza dos dados).

Uso (da raiz do repositório, depois de ``python -m src.common.prepare_data``)::

    python scripts/make_checkpoint2_figures.py

Grava em ``reports/figures/`` (PDF vetorial para o LaTeX e PNG para consulta):

- ``fer2013_cleaning_removals_by_reason``: imagens removidas do treino, por motivo;
- ``fer2013_class_distribution``: distribuição de classes do treino antes e depois da limpeza;
- ``fer2013_std_histogram``: desvio padrão dos pixels por imagem, com o limiar;
- ``degradations_example_face``: as degradações do contrato num rosto de exemplo.

Nenhuma imagem individual da FER-2013 aparece nas figuras. O rosto de exemplo é a
imagem ``astronaut`` do scikit-image (foto da NASA, de domínio público).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common.clean import DEFAULT_STD_THRESHOLD  # noqa: E402
from src.common.data import EMOTIONS, fer2013_arrays, read_clean_csv  # noqa: E402
from src.common.degrade import LEVELS, apply_degradation  # noqa: E402

OUT = REPO_ROOT / "reports" / "figures"

# Cores do tema UFG dos slides. O par azul + laranja escuro passou no validador de paleta
# (faixa de luminosidade, croma, separação para daltonismo e contraste).
BLUE = "#0072B9"      # UFGBlue
ORANGE = "#C76E00"    # DarkOrange do tema, escurecido para ter contraste
INK = "#212121"       # DarkGray (texto)
MUTED = "#6B6B6B"     # texto secundário
GRID = "#E3E3E3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Roboto", "Segoe UI", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"  {OUT / name}.pdf / .png")


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def removals_by_reason(decision: pd.DataFrame) -> None:
    removed = decision[(decision["split"] == "train") & ~decision["keep"]]
    reasons = removed["reason"].str.split(";").explode()
    labels = {
        "dup_within_split": "Duplicata dentro do treino",
        "dup_cross_split_leak": "Cópia de imagem da validação/teste\n(vazamento)",
        "label_conflict": "Cópias com rótulos diferentes",
        "low_std": "Imagem vazia (desvio < 5)",
    }
    counts = reasons.value_counts().reindex(list(labels), fill_value=0)
    order = counts.sort_values().index

    fig, ax = plt.subplots(figsize=(5.0, 2.3))
    y = np.arange(len(order))
    ax.barh(y, counts[order], color=BLUE, height=0.55)
    for yi, key in zip(y, order):
        ax.text(counts[key] + 15, yi, _fmt(int(counts[key])), va="center", fontsize=10, color=INK,
                fontweight="bold")
    ax.set_yticks(y, [labels[k] for k in order], color=INK)
    ax.set_xlim(0, counts.max() * 1.18)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("imagens removidas do treino", color=MUTED)
    ax.set_title(f"{_fmt(len(removed))} imagens removidas (uma imagem pode ter mais de um motivo)",
                 fontsize=10, color=MUTED, loc="left")
    _save(fig, "fer2013_cleaning_removals_by_reason")


def class_distribution(decision: pd.DataFrame) -> None:
    train = decision[decision["split"] == "train"]
    before = train["label"].value_counts(normalize=True).reindex(range(7), fill_value=0) * 100
    after = train.loc[train["keep"], "label"].value_counts(normalize=True).reindex(range(7), fill_value=0) * 100
    n_before, n_after = len(train), int(train["keep"].sum())

    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    x = np.arange(7)
    w = 0.38
    ax.bar(x - w / 2 - 0.01, before, w, color=ORANGE, label=f"treino antes ({_fmt(n_before)})")
    ax.bar(x + w / 2 + 0.01, after, w, color=BLUE, label=f"treino depois ({_fmt(n_after)})")
    for c in (1, 5):  # nojo (a menor classe) e surpresa (a que mais perdeu duplicatas)
        text = f"{before[c]:.1f}% → {after[c]:.1f}%".replace(".", ",")
        ax.annotate(text, (x[c], max(before[c], after[c])),
                    xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8.5, color=INK,
                    bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"), zorder=5)
    ax.set_xticks(x, EMOTIONS, color=INK)
    ax.set_ylabel("% do treino", color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, max(before.max(), after.max()) * 1.18)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _save(fig, "fer2013_class_distribution")


def std_histogram(X: np.ndarray) -> None:
    std = X.reshape(len(X), -1).astype(np.float64).std(axis=1)
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    ax.hist(std, bins=np.arange(0, std.max() + 1, 1.0), color=BLUE)
    ax.axvline(DEFAULT_STD_THRESHOLD, color=ORANGE, ls="--", lw=1.6)
    n_low = int((std < DEFAULT_STD_THRESHOLD).sum())
    ax.text(DEFAULT_STD_THRESHOLD + 1.5, 2500, f"limiar = {DEFAULT_STD_THRESHOLD:g}: {n_low} imagens vazias à esquerda",
            fontsize=9, color=INK, va="center")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 6000)
    ax.set_xlabel("desvio padrão dos pixels de cada imagem", color=MUTED)
    ax.set_ylabel("imagens (escala log)", color=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _save(fig, "fer2013_std_histogram")


def degradations_example() -> None:
    import cv2
    from skimage import data as skdata

    rgb = skdata.astronaut()
    face = cv2.cvtColor(rgb[30:230, 140:340], cv2.COLOR_RGB2GRAY)  # recorte quadrado do rosto
    face = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)[None]

    rows = [("Ruído", "noise", "σ = {}"), ("Desfoque", "blur", "kernel {}"),
            ("Resolução", "resolution", "{} px → 48")]
    fig, axes = plt.subplots(3, 4, figsize=(4.4, 3.5))
    for r, (title, kind, fmt) in enumerate(rows):
        panels = [("original", face[0])] + [(fmt.format(lv), apply_degradation(face, kind, lv)[0])
                                            for lv in LEVELS[kind]]
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(panels):
                name, img = panels[c]
                ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.set_title(name, fontsize=8.5, color=INK if c else MUTED, pad=3)
        axes[r, 0].text(-0.12, 0.5, title, transform=axes[r, 0].transAxes, rotation=90, ha="right",
                        va="center", fontsize=8.5, color=BLUE, fontweight="bold")
    fig.subplots_adjust(wspace=0.06, hspace=0.30)
    _save(fig, "degradations_example_face")


def main() -> int:
    decision = read_clean_csv()
    arr = fer2013_arrays()
    print("Gerando figuras em", OUT)
    removals_by_reason(decision)
    class_distribution(decision)
    std_histogram(arr["X"])
    degradations_example()
    return 0


if __name__ == "__main__":
    sys.exit(main())
