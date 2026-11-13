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

from pathlib import Path
from typing import Literal

import numpy as np

# ------------------------------------------------------------------ constantes
#: Ordem oficial dos rótulos (a mesma da FER-2013). Não mudar.
EMOTIONS: tuple[str, ...] = ("raiva", "nojo", "medo", "alegria", "tristeza", "surpresa", "neutro")
NUM_CLASSES = 7
NEUTRAL = 6
IMG_SIZE = 48

SPLITS: tuple[str, ...] = ("train", "val", "test")
#: Coluna ``Usage`` da FER-2013 -> nome do conjunto no projeto.
USAGE_TO_SPLIT: dict[str, str] = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
#: Tamanhos oficiais da FER-2013 antes da limpeza.
FER2013_SIZES: dict[str, int] = {"train": 28709, "val": 3589, "test": 3589}

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
    raise NotImplementedError


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
    """
    raise NotImplementedError


# ------------------------------------------------------------------------- CK+
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
    raise NotImplementedError


def make_ckplus_folds(cache_path: Path | str | None = None, folds_csv: Path | str | None = None,
                      n_splits: int = CKPLUS_N_FOLDS) -> Path:
    """Gera ``splits/ckplus_folds.csv`` com ``GroupKFold(n_splits)`` agrupado por pessoa.

    Uma linha por sequência, com as colunas ``subject, sequence, label, fold``. A
    sequência faz parte do teste no fold ``fold`` e do treino nos outros. A mesma pessoa
    (``Sxxx``) nunca aparece no treino e no teste do mesmo fold.
    """
    raise NotImplementedError


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
    """
    raise NotImplementedError
