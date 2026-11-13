"""Carregamento da FER-2013 e do CK+ (contrato entre as frentes, CLAUDE.md seção 5).

Os dois carregadores devolvem imagens em escala de cinza ``uint8`` de 48x48 e rótulos
``int64`` na ordem oficial do projeto (ver ``EMOTIONS``). Nenhuma outra parte do código
deve ler os arquivos brutos diretamente.

Uso típico::

    from src.common.data import load_fer2013, load_ckplus

    X_train, y_train, ids_train = load_fer2013("train")       # já limpo
    X_val, y_val, _ = load_fer2013("val")
    X, y, ids = load_ckplus(fold=0, part="train", mode="image")
    seqs, y, ids = load_ckplus(fold=0, part="test", mode="sequence")

Os caminhos padrão são relativos à raiz do repositório, então tudo funciona a partir de
qualquer diretório. Os argumentos só por palavra-chave (``csv_path``, ``cache_path``,
``splits_csv``, ``raw_dir``, ``folds_csv``) existem para os testes e não fazem parte do
contrato.
"""
from __future__ import annotations

import io
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.common.utils import detect_faces

# ------------------------------------------------------------------ constantes
#: Ordem oficial dos rótulos (a mesma da FER-2013). Não mudar.
EMOTIONS: tuple[str, ...] = ("raiva", "nojo", "medo", "alegria", "tristeza", "surpresa", "neutro")
NUM_CLASSES = 7
NEUTRAL = 6
IMG_SIZE = 48

SPLITS: tuple[str, ...] = ("train", "val", "test")
#: Coluna ``Usage`` da FER-2013 -> nome do conjunto no projeto.
USAGE_TO_SPLIT: dict[str, str] = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
#: Tamanhos oficiais da FER-2013 antes da limpeza (por valor de ``Usage``).
FER2013_SIZES: dict[str, int] = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}
FER2013_COLUMNS: tuple[str, ...] = ("emotion", "pixels", "Usage")
N_PIXELS = 48 * 48

#: Rótulos do CK+ (1 raiva, 2 desprezo, 3 nojo, 4 medo, 5 alegria, 6 tristeza,
#: 7 surpresa) -> ordem da FER-2013. O desprezo (2) é descartado.
CKPLUS_TO_FER: dict[int, int] = {1: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5}
CKPLUS_CONTEMPT = 2
CKPLUS_N_FOLDS = 10

Split = Literal["train", "val", "test"]
Part = Literal["train", "test"]
Mode = Literal["image", "sequence"]

# --------------------------------------------------------------------- caminhos
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
SPLITS_DIR = REPO_ROOT / "splits"

FER2013_CSV = DATA_DIR / "raw" / "fer2013" / "fer2013.csv"
FER2013_CACHE = DATA_DIR / "processed" / "fer2013.npz"
FER2013_CLEAN_CSV = SPLITS_DIR / "fer2013_clean.csv"

CKPLUS_RAW_DIR = DATA_DIR / "raw" / "ckplus"
CKPLUS_CACHE = DATA_DIR / "processed" / "ckplus.npz"
CKPLUS_FOLDS_CSV = SPLITS_DIR / "ckplus_folds.csv"


# -------------------------------------------------------------------- FER-2013
def parse_pixels(pixels: Iterable[str], side: int = IMG_SIZE) -> np.ndarray:
    """Converte strings de pixels ("70 80 82 ...") em uint8 [N, side, side].

    Raises:
        ValueError: se alguma linha não tiver ``side * side`` valores ou tiver valor fora
            de 0..255.
    """
    text = "\n".join(pixels)
    if not text:
        return np.zeros((0, side, side), dtype=np.uint8)
    try:
        flat = np.loadtxt(io.StringIO(text), dtype=np.int16, ndmin=2)
    except ValueError as exc:  # linhas com quantidades diferentes de valores, ou texto inválido
        raise ValueError(f"coluna pixels malformada: {exc}") from exc
    if flat.shape[1] != side * side:
        raise ValueError(f"cada linha de pixels deveria ter {side * side} valores, tem {flat.shape[1]}")
    if flat.min() < 0 or flat.max() > 255:
        raise ValueError(f"valores de pixel fora de 0..255 (mín {flat.min()}, máx {flat.max()})")
    return flat.astype(np.uint8).reshape(-1, side, side)


def read_fer2013_csv(csv_path: Path | str | None = None,
                     expected_sizes: Mapping[str, int] | None = FER2013_SIZES
                     ) -> tuple[pd.DataFrame, np.ndarray]:
    """Lê e valida o ``fer2013.csv``. Devolve ``(df, X)`` ou dá ``ValueError`` com a lista de problemas.

    Confere se o arquivo tem 3 colunas (``emotion``, ``pixels``, ``Usage``), cada
    ``pixels`` com 2304 valores entre 0 e 255, ``emotion`` entre 0 e 6, ``Usage`` só com
    Training, PublicTest e PrivateTest e, se ``expected_sizes`` não for ``None``, as
    contagens oficiais 28.709, 3.589 e 3.589 (35.887 no total).
    """
    path = Path(csv_path) if csv_path is not None else FER2013_CSV
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe. Rode primeiro: python scripts/download_data.py")
    df = pd.read_csv(path, dtype={"pixels": str, "Usage": str})
    df.columns = df.columns.str.strip()

    problems: list[str] = []
    if tuple(sorted(df.columns)) != tuple(sorted(FER2013_COLUMNS)):
        raise ValueError(f"colunas esperadas {list(FER2013_COLUMNS)}, encontradas {list(df.columns)}")
    if df.isna().any().any():
        problems.append(f"{int(df.isna().any(axis=1).sum())} linhas com valores vazios")
    df = df.dropna()
    if expected_sizes is not None and len(df) != sum(expected_sizes.values()):
        problems.append(f"{len(df)} linhas, esperado {sum(expected_sizes.values())}")
    n_values = df["pixels"].str.strip().str.count(" ") + 1
    bad = np.flatnonzero(n_values.to_numpy() != N_PIXELS)
    if bad.size:
        problems.append(f"{bad.size} linhas sem {N_PIXELS} pixels (ex.: linha {int(bad[0])} "
                        f"tem {int(n_values.iloc[bad[0]])})")
    if not pd.api.types.is_integer_dtype(df["emotion"]) or not df["emotion"].between(0, NUM_CLASSES - 1).all():
        problems.append("emotion fora de 0..6 ou não inteiro")
    usage = set(df["Usage"].unique())
    if not usage <= set(USAGE_TO_SPLIT):
        problems.append(f"valores de Usage inesperados: {sorted(usage - set(USAGE_TO_SPLIT))}")
    if expected_sizes is not None:
        counts = df["Usage"].value_counts().to_dict()
        for usage_name, n in expected_sizes.items():
            if counts.get(usage_name, 0) != n:
                problems.append(f"Usage={usage_name}: {counts.get(usage_name, 0)} linhas, esperado {n}")
    if problems:
        raise ValueError(f"{path} inválido:\n  - " + "\n  - ".join(problems))

    X = parse_pixels(df["pixels"])
    df = df[list(FER2013_COLUMNS)].reset_index(drop=True)
    df["emotion"] = df["emotion"].astype(np.int64)
    return df, X


def fer2013_arrays(csv_path: Path | str | None = None, cache_path: Path | str | None = None,
                   force: bool = False) -> dict[str, np.ndarray]:
    """Todas as 35.887 imagens da FER-2013, sem limpeza, com cache em ``.npz``.

    Lê ``data/raw/fer2013/fer2013.csv`` uma vez e guarda o resultado em
    ``data/processed/fer2013.npz``. O cache é refeito se o CSV mudar (tamanho ou data
    de modificação) ou com ``force=True``.

    Returns:
        Dicionário com
        ``X`` (uint8 [N, 48, 48]), ``y`` (int64 [N]), ``split`` (str [N], em
        ``SPLITS``) e ``ids`` (int64 [N], índice da linha no CSV original).
    """
    csv = Path(csv_path) if csv_path is not None else FER2013_CSV
    cache = Path(cache_path) if cache_path is not None else FER2013_CACHE
    if not csv.is_file():
        raise FileNotFoundError(f"{csv} não existe. Rode primeiro: python scripts/download_data.py")
    st = csv.stat()
    stamp = np.array([st.st_size, st.st_mtime_ns], dtype=np.int64)
    keys = ("X", "y", "split", "ids")

    if cache.is_file() and not force:
        with np.load(cache, allow_pickle=False) as z:
            if "source_stamp" in z and np.array_equal(z["source_stamp"], stamp):
                return {k: z[k] for k in keys}

    # A estrutura é validada aqui. As contagens oficiais são conferidas pelo download e pelo prepare_data.
    df, X = read_fer2013_csv(csv, expected_sizes=None)
    out = {
        "X": X,
        "y": df["emotion"].to_numpy(np.int64),
        "split": df["Usage"].map(USAGE_TO_SPLIT).to_numpy().astype("<U5"),
        "ids": np.arange(len(df), dtype=np.int64),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, source_stamp=stamp, **out)
    tmp.replace(cache)
    return out


def read_clean_csv(splits_csv: Path | str | None = None) -> pd.DataFrame:
    """Lê ``splits/fer2013_clean.csv`` (colunas ``id, split, label, keep, reason``)."""
    path = Path(splits_csv) if splits_csv is not None else FER2013_CLEAN_CSV
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe. Rode: python -m src.common.prepare_data "
                                f"(ou use load_fer2013(split, clean=False) para os dados sem limpeza)")
    df = pd.read_csv(path, dtype={"id": np.int64, "split": str, "label": np.int64, "keep": bool, "reason": str},
                     keep_default_na=False)
    missing = {"id", "split", "label", "keep", "reason"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: faltam as colunas {sorted(missing)}")
    return df


def load_fer2013(split: Split, clean: bool = True, *, csv_path: Path | str | None = None,
                 cache_path: Path | str | None = None,
                 splits_csv: Path | str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carrega um conjunto da FER-2013.

    Args:
        split: ``"train"`` (Training), ``"val"`` (PublicTest) ou ``"test"`` (PrivateTest).
        clean: se ``True`` (padrão), mantém só as linhas com ``keep=True`` em
            ``splits/fer2013_clean.csv``. A limpeza só remove imagens do treino; val e
            teste ficam sempre completos (3.589 imagens cada).

    Returns:
        ``(X, y, ids)``:
        ``X`` uint8 [N, 48, 48], ``y`` int64 [N] com rótulos 0..6 na ordem de
        ``EMOTIONS`` e ``ids`` int64 [N] com o índice da linha no CSV original
        (identificador estável, o mesmo de ``splits/fer2013_clean.csv``).

    Raises:
        FileNotFoundError: se o CSV não existir (rode ``python scripts/download_data.py``)
            ou, com ``clean=True``, se ``splits/fer2013_clean.csv`` não existir
            (rode ``python -m src.common.prepare_data``).
        ValueError: se o arquivo de limpeza não corresponder ao CSV (outra versão dos dados).
    """
    if split not in SPLITS:
        raise ValueError(f"split deve ser um de {SPLITS}, não {split!r}")
    arr = fer2013_arrays(csv_path, cache_path)
    mask = arr["split"] == split
    if clean:
        decision = read_clean_csv(splits_csv).sort_values("id")
        ids_ok = len(decision) == len(arr["ids"]) and np.array_equal(decision["id"].to_numpy(), arr["ids"])
        if not ids_ok or not np.array_equal(decision["label"].to_numpy(), arr["y"]) \
                or not np.array_equal(decision["split"].to_numpy(), arr["split"]):
            raise ValueError("splits/fer2013_clean.csv não corresponde ao fer2013.csv desta máquina "
                             "(ids, rótulos ou splits diferentes). Confira o SHA-256 em "
                             "data/raw/fer2013/CHECKSUM.txt e rode python -m src.common.prepare_data.")
        mask &= decision["keep"].to_numpy(bool)
    return arr["X"][mask], arr["y"][mask], arr["ids"][mask]


# ------------------------------------------------------------------------- CK+
CKPLUS_IMAGES_DIRNAME = "cohn-kanade-images"
CKPLUS_EMOTION_DIRNAME = "Emotion"


def find_ckplus_dirs(raw_dir: Path | str | None = None, max_depth: int = 3) -> tuple[Path, Path] | None:
    """Procura ``cohn-kanade-images/`` e ``Emotion/`` até ``max_depth`` níveis abaixo de ``raw_dir``.

    Alguns zips do CK+ trazem as pastas aninhadas (ex.: ``Emotion_labels/Emotion/``). A
    busca ignora ``__MACOSX``.

    Returns:
        ``(images_dir, emotion_dir)`` ou ``None`` se alguma das duas faltar.
    """
    root = Path(raw_dir) if raw_dir is not None else CKPLUS_RAW_DIR
    found: dict[str, Path] = {}
    frontier = [root] if root.is_dir() else []
    for _ in range(max_depth + 1):
        next_frontier: list[Path] = []
        for d in frontier:
            if d.name in (CKPLUS_IMAGES_DIRNAME, CKPLUS_EMOTION_DIRNAME):
                found.setdefault(d.name, d)
                continue
            next_frontier += sorted(p for p in d.iterdir() if p.is_dir() and p.name != "__MACOSX")
        frontier = next_frontier
    if CKPLUS_IMAGES_DIRNAME in found and CKPLUS_EMOTION_DIRNAME in found:
        return found[CKPLUS_IMAGES_DIRNAME], found[CKPLUS_EMOTION_DIRNAME]
    return None


def build_ckplus_cache(raw_dir: Path | str | None = None, cache_path: Path | str | None = None,
                       force: bool = False) -> Path:
    """Lê as sequências rotuladas do CK+, recorta o rosto e grava o cache ``.npz``.

    Estrutura esperada em ``raw_dir`` (padrão ``data/raw/ckplus/``)::

        cohn-kanade-images/Sxxx/yyy/Sxxx_yyy_zzzzzzzz.png
        Emotion/Sxxx/yyy/Sxxx_yyy_zzzzzzzz_emotion.txt

    Só entram as sequências com rótulo. O desprezo (2) é descartado e o resto é mapeado
    por ``CKPLUS_TO_FER``. O rosto é detectado com Viola-Jones no primeiro quadro (o
    maior rosto), e a mesma caixa é usada na sequência inteira. Depois disso, cada
    quadro vai para cinza e é redimensionado para 48x48. Se nenhum rosto for detectado,
    usa um recorte quadrado central e registra isso no cache.

    Returns:
        Caminho do cache (padrão ``data/processed/ckplus.npz``).
    """
    cache = Path(cache_path) if cache_path is not None else CKPLUS_CACHE
    if cache.is_file() and not force:
        return cache
    root = Path(raw_dir) if raw_dir is not None else CKPLUS_RAW_DIR
    dirs = find_ckplus_dirs(root)
    if dirs is None:
        raise FileNotFoundError(
            f"CK+ não encontrado em {root} (faltam cohn-kanade-images/ e Emotion/). O acesso é pedido aos "
            f"autores; quando os zips chegarem, coloque em data/raw/ckplus/ e rode "
            f"python scripts/download_data.py --only ckplus. Veja data/raw/ckplus/LEIA-ME.txt.")
    images_dir, emotion_dir = dirs

    frames_all: list[np.ndarray] = []
    offsets = [0]
    labels: list[int] = []
    subjects: list[str] = []
    sequences: list[str] = []
    detected: list[bool] = []
    boxes: list[tuple[int, int, int, int]] = []
    skipped = {"desprezo": 0, "rótulo desconhecido": 0, "sem quadros": 0}

    label_files = sorted(p for p in emotion_dir.rglob("*_emotion.txt") if not p.name.startswith("."))
    seen: set[tuple[str, str]] = set()
    for label_file in label_files:
        subject, seq = label_file.parent.parent.name, label_file.parent.name
        if (subject, seq) in seen:  # mais de um arquivo de rótulo na mesma sequência: fica o primeiro
            continue
        seen.add((subject, seq))
        ck_label = int(round(float(label_file.read_text().split()[0])))
        if ck_label == CKPLUS_CONTEMPT:
            skipped["desprezo"] += 1
            continue
        if ck_label not in CKPLUS_TO_FER:
            skipped["rótulo desconhecido"] += 1
            continue
        frame_paths = _ckplus_frame_paths(images_dir / subject / seq)
        if not frame_paths:
            skipped["sem quadros"] += 1
            continue
        raw = [_read_gray(p) for p in frame_paths]
        box, ok = _face_box(raw[0])
        frames_all.append(np.stack([_crop_resize(f, box) for f in raw]))
        offsets.append(offsets[-1] + len(raw))
        labels.append(CKPLUS_TO_FER[ck_label])
        subjects.append(subject)
        sequences.append(seq)
        detected.append(ok)
        boxes.append(box)

    if not labels:
        raise ValueError(f"nenhuma sequência rotulada utilizável em {emotion_dir} (descartadas: {skipped})")
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".part")
    with open(tmp, "wb") as f:
        np.savez(f, frames=np.concatenate(frames_all), offsets=np.asarray(offsets, np.int64),
                 labels=np.asarray(labels, np.int64), subjects=np.asarray(subjects, dtype="<U16"),
                 sequences=np.asarray(sequences, dtype="<U16"), face_detected=np.asarray(detected),
                 boxes=np.asarray(boxes, np.int64),
                 skipped=np.asarray([skipped[k] for k in skipped], np.int64),
                 skipped_keys=np.asarray(list(skipped), dtype="<U32"))
    tmp.replace(cache)
    return cache


def _ckplus_frame_paths(seq_dir: Path) -> list[Path]:
    if not seq_dir.is_dir():
        return []
    return sorted(p for p in seq_dir.iterdir()
                  if p.suffix.lower() == ".png" and not p.name.startswith("."))


def _read_gray(path: Path) -> np.ndarray:
    """Lê uma imagem em cinza uint8. Usa ``imdecode`` para funcionar com caminhos não ASCII no Windows."""
    import cv2

    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"não consegui ler {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:  # por segurança: imagens de 16 bits viram 8 bits
        img = (img.astype(np.float64) * 255.0 / max(int(img.max()), 1)).round().astype(np.uint8)
    return img


def _face_box(gray: np.ndarray) -> tuple[tuple[int, int, int, int], bool]:
    """Maior rosto do Viola-Jones como (x, y, w, h). Sem rosto, devolve o quadrado central e ``False``."""
    h, w = gray.shape
    side_min = max(min(h, w) // 5, 20)
    faces = detect_faces(gray, scale_factor=1.1, min_neighbors=5, min_size=(side_min, side_min))
    if len(faces):
        x, y, fw, fh = (int(v) for v in faces[np.argmax(faces[:, 2] * faces[:, 3])])
        return (x, y, fw, fh), True
    side = min(h, w)
    return ((w - side) // 2, (h - side) // 2, side, side), False


def _crop_resize(gray: np.ndarray, box: tuple[int, int, int, int], size: int = IMG_SIZE) -> np.ndarray:
    import cv2

    x, y, w, h = box
    H, W = gray.shape
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError(f"caixa {box} fora da imagem {gray.shape}")
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def _load_ckplus_cache(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_ckplus_folds(folds_csv: Path | str | None = None) -> pd.DataFrame:
    """Lê ``splits/ckplus_folds.csv`` (``subject, sequence, label, fold``)."""
    path = Path(folds_csv) if folds_csv is not None else CKPLUS_FOLDS_CSV
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe. Rode: python -m src.common.prepare_data "
                                f"(depois de colocar o CK+ em data/raw/ckplus/)")
    return pd.read_csv(path, dtype={"subject": str, "sequence": str, "label": np.int64, "fold": np.int64})


def make_ckplus_folds(cache_path: Path | str | None = None, folds_csv: Path | str | None = None,
                      n_splits: int = CKPLUS_N_FOLDS) -> Path:
    """Gera ``splits/ckplus_folds.csv`` com ``GroupKFold(n_splits)`` agrupado por pessoa.

    Uma linha por sequência, com as colunas ``subject, sequence, label, fold``. A
    sequência faz parte do teste no fold ``fold`` e do treino nos outros. A mesma pessoa
    (``Sxxx``) nunca aparece no treino e no teste do mesmo fold. O ``label`` já está na
    ordem da FER-2013. O ``GroupKFold`` sem embaralhamento é determinístico.
    """
    from sklearn.model_selection import GroupKFold

    cache = Path(cache_path) if cache_path is not None else CKPLUS_CACHE
    if not cache.is_file():
        raise FileNotFoundError(f"{cache} não existe; rode build_ckplus_cache() antes")
    c = _load_ckplus_cache(cache)
    subjects, labels = c["subjects"], c["labels"]
    n_subjects = len(np.unique(subjects))
    if n_subjects < n_splits:
        raise ValueError(f"só {n_subjects} pessoas para {n_splits} folds")
    fold = np.full(len(labels), -1, dtype=np.int64)
    for k, (_, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(labels)), labels, subjects)):
        fold[test_idx] = k
    df = pd.DataFrame({"subject": subjects, "sequence": c["sequences"], "label": labels, "fold": fold})
    df = df.sort_values(["subject", "sequence"]).reset_index(drop=True)
    path = Path(folds_csv) if folds_csv is not None else CKPLUS_FOLDS_CSV
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def load_ckplus(fold: int, part: Part = "train", mode: Mode = "image", *,
                raw_dir: Path | str | None = None, cache_path: Path | str | None = None,
                folds_csv: Path | str | None = None
                ) -> tuple[np.ndarray | list[np.ndarray], np.ndarray, np.ndarray]:
    """Carrega o treino ou o teste de um fold do CK+ (validação cruzada por pessoa).

    Args:
        fold: 0..9. O teste é a parte ``fold`` de ``splits/ckplus_folds.csv``, e o treino
            é o resto.
        part: ``"train"`` ou ``"test"``.
        mode:
            ``"image"``: duas imagens por sequência, o último quadro (pico, com o
            rótulo da emoção) e o primeiro quadro (rótulo 6, neutro).
            ``"sequence"``: todos os quadros, do neutro ao pico, com o rótulo da
            emoção (nesse modo não existe a classe neutro).

    Returns:
        ``(X, y, ids)``:
        ``X`` é uint8 [N, 48, 48] no modo ``"image"`` e uma lista de N arrays uint8
        [T_i, 48, 48] no modo ``"sequence"``. ``y`` é int64 [N] na ordem de
        ``EMOTIONS``. ``ids`` são strings ``"S005/001/peak"`` ou ``"S005/001/neutral"``
        (modo image) e ``"S005/001"`` (modo sequence).

    Raises:
        FileNotFoundError: se o CK+ ainda não estiver em ``data/raw/ckplus/`` ou se
            ``splits/ckplus_folds.csv`` não existir.
    """
    if part not in ("train", "test"):
        raise ValueError(f"part deve ser 'train' ou 'test', não {part!r}")
    if mode not in ("image", "sequence"):
        raise ValueError(f"mode deve ser 'image' ou 'sequence', não {mode!r}")
    cache = build_ckplus_cache(raw_dir, cache_path)  # só monta se o cache ainda não existir
    c = _load_ckplus_cache(cache)
    folds = read_ckplus_folds(folds_csv)
    n_folds = int(folds["fold"].max()) + 1
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold deve estar em 0..{n_folds - 1}, não {fold}")

    keys = np.char.add(np.char.add(c["subjects"], "/"), c["sequences"])
    fold_of = dict(zip(folds["subject"] + "/" + folds["sequence"], folds["fold"]))
    if set(fold_of) != set(keys.tolist()):
        raise ValueError("splits/ckplus_folds.csv não corresponde ao cache do CK+ (sequências diferentes). "
                         "Rode python -m src.common.prepare_data --force.")
    seq_fold = np.array([fold_of[k] for k in keys], dtype=np.int64)
    selected = np.flatnonzero((seq_fold == fold) == (part == "test"))

    frames, offsets, labels = c["frames"], c["offsets"], c["labels"]
    if mode == "sequence":
        seqs = [frames[offsets[i]:offsets[i + 1]].copy() for i in selected]
        return seqs, labels[selected].astype(np.int64), keys[selected].astype(object)

    X = np.empty((2 * len(selected), IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    y = np.empty(2 * len(selected), dtype=np.int64)
    ids = np.empty(2 * len(selected), dtype=object)
    for j, i in enumerate(selected):
        X[2 * j], y[2 * j], ids[2 * j] = frames[offsets[i + 1] - 1], labels[i], f"{keys[i]}/peak"
        X[2 * j + 1], y[2 * j + 1], ids[2 * j + 1] = frames[offsets[i]], NEUTRAL, f"{keys[i]}/neutral"
    return X, y, ids
