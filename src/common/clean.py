"""Limpeza da FER-2013 e pesos de classe.

Nada é apagado. A limpeza gera um arquivo de decisão, ``splits/fer2013_clean.csv``
(versionado), com uma linha por imagem e as colunas ``id, split, label, keep, reason``,
e um relatório, ``reports/data_cleaning.md``.

Regras (só o treino perde imagens, e a validação e o teste nunca são alterados):

=========================  ==================================  ============================
Situação                   Treino                              Validação / teste
=========================  ==================================  ============================
Duplicata no mesmo split   fica a de menor id, as outras saem  ``flag:dup_within_split``
                           (``dup_within_split``)
Duplicata treino x val/te  sai do treino                       intacta
                           (``dup_cross_split_leak``)
Duplicata val x teste      não se aplica                       ``flag:dup_val_test``
Rótulos conflitantes       todas as cópias saem                ``flag:label_conflict``
                           (``label_conflict``)
Quase constante (std < t)  sai (``low_std``)                   ``flag:low_std``
Viola-Jones sem rosto      ``flag:no_face`` (não sai)          ``flag:no_face``
=========================  ==================================  ============================

Quando há mais de um motivo, eles vêm separados por ``;``. Os motivos com prefixo
``flag:`` só sinalizam e nunca removem.
"""
from __future__ import annotations

import hashlib
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data import EMOTIONS, FER2013_CLEAN_CSV, NUM_CLASSES, REPO_ROOT, SPLITS

#: Motivos que removem a imagem do treino.
REMOVE_REASONS: tuple[str, ...] = ("dup_within_split", "dup_cross_split_leak", "label_conflict", "low_std")
#: Motivos que só sinalizam.
FLAG_REASONS: tuple[str, ...] = ("flag:dup_within_split", "flag:dup_val_test", "flag:label_conflict",
                                 "flag:low_std", "flag:no_face")
DEFAULT_STD_THRESHOLD = 5.0
CLEAN_COLUMNS: tuple[str, ...] = ("id", "split", "label", "keep", "reason")
REPORT_PATH = REPO_ROOT / "reports" / "data_cleaning.md"
INSPECTION_DIR = REPO_ROOT / "data" / "processed" / "inspection"
_ORDER = {r: i for i, r in enumerate(REMOVE_REASONS + FLAG_REASONS)}


def pixel_hashes(X: np.ndarray) -> np.ndarray:
    """MD5 (hex) dos bytes de cada imagem uint8 [N, H, W]."""
    X = np.ascontiguousarray(X, dtype=np.uint8)
    return np.array([hashlib.md5(img.tobytes()).hexdigest() for img in X], dtype="<U32")


def detect_no_face(X: np.ndarray, upscale: int = 96, min_size: int = 24) -> np.ndarray:
    """Marca as imagens em que o Viola-Jones não encontra rosto (bool [N]).

    A imagem é ampliada para ``upscale`` x ``upscale`` antes da detecção. Em 48x48 o
    detector erra bastante, por isso o resultado só serve como sinalização.
    """
    import cv2
    from tqdm import tqdm

    from src.common.utils import detect_faces

    out = np.zeros(len(X), dtype=bool)
    for i, img in enumerate(tqdm(X, desc="Viola-Jones (sinalização)", unit="img", leave=False)):
        big = cv2.resize(img, (upscale, upscale), interpolation=cv2.INTER_LINEAR)
        out[i] = len(detect_faces(big, min_neighbors=3, min_size=(min_size, min_size))) == 0
    return out


def clean_fer2013(X: np.ndarray, y: np.ndarray, split: np.ndarray, ids: np.ndarray, *,
                  std_threshold: float = DEFAULT_STD_THRESHOLD, face_check: bool = True) -> pd.DataFrame:
    """Aplica as regras do módulo e devolve a tabela de decisão.

    Args:
        X, y, split, ids: saída de ``src.common.data.fer2013_arrays()`` (todas as imagens).
        std_threshold: imagens com desvio padrão dos pixels abaixo disto são "quase
            constantes".
        face_check: roda o Viola-Jones para sinalizar ``flag:no_face``.

    Returns:
        DataFrame com as colunas ``CLEAN_COLUMNS``, mais ``std`` e ``md5`` (que servem
        ao relatório e não vão para o CSV versionado), ordenado por ``id``.
    """
    n = len(X)
    y = np.asarray(y, dtype=np.int64)
    split = np.asarray(split).astype(str)
    ids = np.asarray(ids, dtype=np.int64)
    if not (len(y) == len(split) == len(ids) == n):
        raise ValueError("X, y, split e ids precisam ter o mesmo tamanho")
    if not set(np.unique(split)) <= set(SPLITS):
        raise ValueError(f"split com valores fora de {SPLITS}")

    reasons: list[set[str]] = [set() for _ in range(n)]
    md5 = pixel_hashes(X)
    std = X.reshape(n, -1).astype(np.float64).std(axis=1)

    # ---- duplicatas exatas, agrupadas pelo hash
    table = pd.DataFrame({"pos": np.arange(n), "md5": md5, "split": split, "label": y, "id": ids})
    dups = table[table.duplicated("md5", keep=False)]
    for _, g in dups.groupby("md5", sort=False):
        train = g[g["split"] == "train"].sort_values("id")
        evals = g[g["split"] != "train"]
        conflict = g["label"].nunique() > 1
        if conflict:
            for p in train["pos"]:
                reasons[p].add("label_conflict")
            for p in evals["pos"]:
                reasons[p].add("flag:label_conflict")
        if len(evals) and len(train):
            for p in train["pos"]:
                reasons[p].add("dup_cross_split_leak")
        if not conflict and not len(evals) and len(train) > 1:
            for p in train["pos"].iloc[1:]:  # fica a cópia de menor id
                reasons[p].add("dup_within_split")
        for s in ("val", "test"):
            same = evals[evals["split"] == s]
            if len(same) > 1:
                for p in same["pos"]:
                    reasons[p].add("flag:dup_within_split")
        if evals["split"].nunique() > 1:
            for p in evals["pos"]:
                reasons[p].add("flag:dup_val_test")

    # ---- imagens quase constantes
    for p in np.flatnonzero(std < std_threshold):
        reasons[p].add("low_std" if split[p] == "train" else "flag:low_std")

    # ---- provável sem rosto (só sinalização)
    if face_check:
        for p in np.flatnonzero(detect_no_face(X)):
            reasons[p].add("flag:no_face")

    keep = np.array([not (r & set(REMOVE_REASONS)) for r in reasons])
    assert keep[split != "train"].all(), "a limpeza nunca pode remover imagens de val/teste"
    df = pd.DataFrame({
        "id": ids,
        "split": split,
        "label": y,
        "keep": keep,
        "reason": [";".join(sorted(r, key=_ORDER.__getitem__)) for r in reasons],
        "std": std,
        "md5": md5,
    })
    return df.sort_values("id").reset_index(drop=True)


def write_clean_csv(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Grava ``splits/fer2013_clean.csv`` (só as colunas ``CLEAN_COLUMNS``)."""
    path = Path(path) if path is not None else FER2013_CLEAN_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df[list(CLEAN_COLUMNS)].to_csv(path, index=False, lineterminator="\n")
    return path


def save_low_std_grid(X: np.ndarray, df: pd.DataFrame, path: Path | str | None = None,
                      std_threshold: float = DEFAULT_STD_THRESHOLD, max_images: int = 96) -> Path:
    """Grade das imagens quase constantes e das que estão perto do limiar, para conferência visual.

    A parte de cima mostra as imagens marcadas (std < t) e a de baixo, as que ficam logo
    acima do limiar (t <= std < 2t). Os exemplos de cada parte são espaçados por igual ao
    longo do std. Também grava um histograma do std ao lado
    (``*_hist.png``). O padrão é ``data/processed/inspection/fer2013_low_std_grid.png``,
    que fica fora do git porque contém imagens da base.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path) if path is not None else INSPECTION_DIR / "fer2013_low_std_grid.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(df) != len(X):
        raise ValueError("df e X precisam estar alinhados (uma linha por imagem, na mesma ordem)")
    std = df["std"].to_numpy()
    cols = 12
    blocks = []
    for title, mask in ((f"marcadas: std < {std_threshold:g}", std < std_threshold),
                        (f"logo acima do limiar: {std_threshold:g} <= std < {2 * std_threshold:g}",
                         (std >= std_threshold) & (std < 2 * std_threshold))):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(std[idx])]
        total = len(idx)
        if total > max_images // 2:
            idx = idx[np.unique(np.linspace(0, total - 1, max_images // 2).round().astype(int))]
        blocks.append((f"{title} ({total} imagens, mostrando {len(idx)})", idx))

    rows_per_block = [max(1, int(np.ceil(len(idx) / cols))) for _, idx in blocks]
    fig = plt.figure(figsize=(cols * 1.1, sum(rows_per_block) * 1.35 + 1.2))
    sub = fig.subfigures(len(blocks), 1, height_ratios=rows_per_block)
    for sf, (title, idx), nrows in zip(np.atleast_1d(sub), blocks, rows_per_block):
        sf.suptitle(title, fontsize=10)
        axes = sf.subplots(nrows, cols, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, k in zip(axes.ravel(), idx):
            ax.imshow(X[k], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{df['split'].iloc[k]} #{df['id'].iloc[k]}\nstd={std[k]:.1f}", fontsize=6)
        if not len(idx):
            axes[0, 0].text(0, 0.5, "nenhuma imagem", fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(std, bins=120, color="#4c72b0")
    ax.axvline(std_threshold, color="#c44e52", ls="--", label=f"limiar = {std_threshold:g}")
    ax.set_yscale("log")
    ax.set_xlabel("desvio padrão dos pixels")
    ax.set_ylabel("imagens (escala log)")
    ax.set_title("FER-2013: desvio padrão por imagem")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path.with_name(path.stem.replace("_grid", "") + "_hist.png"), dpi=110)
    plt.close(fig)
    return path


def class_weights(y: np.ndarray, n_classes: int = NUM_CLASSES) -> np.ndarray:
    """Pesos de classe pelo inverso da frequência, normalizados (esquema "balanced").

    ``w_c = N / (n_classes * n_c)``, igual ao ``class_weight="balanced"`` do
    scikit-learn. A média dos pesos ponderada pela frequência é 1. Serve para
    ``CrossEntropyLoss(weight=...)`` e para os classificadores da Frente B. A classe
    "nojo" (1) é a menor e recebe o maior peso. Uma classe ausente recebe peso 0.

    Returns:
        float64 [n_classes].
    """
    y = np.asarray(y).astype(np.int64).ravel()
    if y.size == 0:
        raise ValueError("y vazio")
    if y.min() < 0 or y.max() >= n_classes:
        raise ValueError(f"rótulos fora de 0..{n_classes - 1}")
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    weights[present] = y.size / (n_classes * counts[present])
    if not present.all():
        warnings.warn(f"classes ausentes recebem peso 0: {np.flatnonzero(~present).tolist()}", stacklevel=2)
    return weights


# --------------------------------------------------------------------- relatório
def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _reason_counts(df: pd.DataFrame, reasons: tuple[str, ...], by: str) -> pd.DataFrame:
    exploded = df.assign(r=df["reason"].str.split(";")).explode("r")
    exploded = exploded[exploded["r"].isin(reasons)]
    if exploded.empty:
        return pd.DataFrame(index=pd.Index(list(reasons), name="r"))
    return exploded.groupby(["r", by]).size().unstack(fill_value=0).reindex(index=list(reasons), fill_value=0)


def write_report(df: pd.DataFrame, path: Path | str | None = None, *, std_threshold: float,
                 csv_sha256: str | None = None, face_check: bool = True,
                 grid_path: Path | str | None = None) -> Path:
    """Gera ``reports/data_cleaning.md`` com as contagens por motivo e classe e as distribuições."""
    path = Path(path) if path is not None else REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = [f"{i} {name}" for i, name in enumerate(EMOTIONS)]
    train = df[df["split"] == "train"]
    removed = train[~train["keep"]]

    # resumo por split
    summary = pd.DataFrame({
        "conjunto": list(SPLITS),
        "antes": [int((df["split"] == s).sum()) for s in SPLITS],
        "removidas": [int(((df["split"] == s) & ~df["keep"]).sum()) for s in SPLITS],
        "depois": [int(((df["split"] == s) & df["keep"]).sum()) for s in SPLITS],
        "sinalizadas (ficam)": [int(((df["split"] == s) & df["keep"] & (df["reason"] != "")).sum()) for s in SPLITS],
    })

    # remoções no treino: motivo x classe
    by_reason = _reason_counts(removed, REMOVE_REASONS, "label").reindex(columns=range(NUM_CLASSES), fill_value=0)
    by_reason.columns = classes
    by_reason["total"] = by_reason.sum(axis=1)
    by_reason.loc["imagens removidas (únicas)"] = (
        [int((removed["label"] == c).sum()) for c in range(NUM_CLASSES)] + [len(removed)])
    by_reason = by_reason.reset_index().rename(columns={"index": "motivo", "r": "motivo"})

    # sinalizações: motivo x split
    flags = _reason_counts(df, FLAG_REASONS, "split").reindex(columns=list(SPLITS), fill_value=0)
    flags = flags.reset_index().rename(columns={"r": "sinalização"})

    # distribuição de classes antes e depois
    def dist(mask: pd.Series) -> list[str]:
        counts = df.loc[mask, "label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        total = max(int(counts.sum()), 1)
        return [f"{int(c)} ({100 * c / total:.1f}%)" for c in counts]

    distribution = pd.DataFrame({
        "classe": classes,
        "treino antes": dist(df["split"] == "train"),
        "treino depois": dist((df["split"] == "train") & df["keep"]),
        "val": dist(df["split"] == "val"),
        "teste": dist(df["split"] == "test"),
    })
    weights = class_weights(train.loc[train["keep"], "label"].to_numpy())
    weights_table = pd.DataFrame({"classe": classes, "peso": [f"{w:.3f}" for w in weights]})

    n_groups = int(df.loc[df.duplicated("md5", keep=False), "md5"].nunique()) if "md5" in df else 0
    grid_rel = None
    if grid_path:
        grid_abs = Path(grid_path).resolve()
        grid_rel = (grid_abs.relative_to(REPO_ROOT) if grid_abs.is_relative_to(REPO_ROOT) else grid_abs).as_posix()
    no_face_total = int(df["reason"].str.contains("flag:no_face").sum())
    text = f"""# Limpeza da FER-2013

> Arquivo gerado por `python -m src.common.prepare_data` em
> {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. Não edite à mão: rode o comando de novo.

- **Arquivo de origem:** `data/raw/fer2013/fer2013.csv`, SHA-256 `{csv_sha256 or 'não calculado'}`
- **Decisão por imagem:** [`splits/fer2013_clean.csv`](../splits/fer2013_clean.csv) (colunas `id, split, label, keep, reason`)
- **Limiar de imagem quase constante:** desvio padrão dos pixels < **{std_threshold:g}**
- **Sinalização por Viola-Jones:** {'sim' if face_check else 'não (rodado com --skip-face-check)'}
- **Grupos de imagens idênticas (MD5 dos pixels):** {n_groups}

**Regras.** Só o treino perde imagens. A validação (PublicTest) e o teste (PrivateTest) ficam
intactos, para os números continuarem comparáveis com a literatura. Neles, os problemas
são só sinalizados (`flag:...`). Nenhum arquivo é apagado: quem decide é a coluna `keep`.

| motivo | o que é | treino | val / teste |
|---|---|---|---|
| `dup_within_split` | cópia exata de outra imagem do treino com o mesmo rótulo | sai (fica a de menor `id`) | `flag:dup_within_split` |
| `dup_cross_split_leak` | a imagem do treino também está em val ou teste (vazamento) | sai | intacta |
| `label_conflict` | cópias idênticas com rótulos diferentes | todas as cópias saem | `flag:label_conflict` |
| `low_std` | imagem quase constante (vazia, preta, branca) | sai | `flag:low_std` |
| `flag:dup_val_test` | a mesma imagem em val e em teste | não se aplica | só sinaliza |
| `flag:no_face` | o Viola-Jones não achou rosto (ampliada para 96x96) | só sinaliza | só sinaliza |

## Resumo por conjunto

{_md_table(summary)}

## Imagens removidas do treino, por motivo e classe

Uma imagem pode ter mais de um motivo. A última linha conta cada imagem uma vez só.

{_md_table(by_reason)}

## Sinalizações (as imagens ficam), por conjunto

{_md_table(flags)}

O Viola-Jones sinalizou {no_face_total} imagens no total. Em 48x48 o detector falha muito,
até em rostos bons, então esse número **não** mede a qualidade da base e nunca é usado para
remover imagens.

## Distribuição de classes antes e depois

{_md_table(distribution)}

## Pesos de classe do treino limpo

`src.common.clean.class_weights(y)`: inverso da frequência normalizado,
`N / (7 · n_c)`, igual ao `class_weight="balanced"` do scikit-learn. A classe com
menos exemplos (nojo) recebe o maior peso.

{_md_table(weights_table)}

## Conferência visual do limiar

"""
    if grid_rel:
        text += (f"A grade com as imagens marcadas e as que estão logo acima do limiar, e o histograma\n"
                 f"do desvio padrão, ficam em `{grid_rel}` e no `_hist.png` ao lado. Esses arquivos\n"
                 f"não são versionados porque contêm imagens da base. Para trocar o limiar:\n"
                 f"`python -m src.common.prepare_data --std-threshold <valor>`.\n")
    else:
        text += "A grade não foi gerada nesta execução.\n"
    path.write_text(text, encoding="utf-8")
    return path
