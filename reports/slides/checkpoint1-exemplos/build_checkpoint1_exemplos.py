"""Insere no checkpoint 1 (PDF exportado do Canva) um slide com exemplos reais da FER-2013.

A inserção é cirúrgica:
- as páginas originais ficam intactas, exceto o número do rodapé ("n/9" vira "n/10"), trocado
  por redação só do texto, na mesma fonte, tamanho, cor e posição;
- o slide novo é uma cópia do slide "2. Contexto: emoções básicas e bases de dados" (mesmo fundo e
  logo), com o conteúdo trocado por uma galeria das 7 emoções no mesmo padrão tipográfico do deck
  (Open Sans nos títulos e rótulos, IBM Plex Sans na nota).

As imagens são escolhidas à mão (ids fixos em ``EXAMPLES``) entre as do treino que passaram na
limpeza sem nenhum aviso (``keep=True`` e ``reason`` vazio em ``splits/fer2013_clean.csv``).
O PDF gerado contém imagens da base e por isso não é versionado; só este script é.

Uso (da raiz do repositório, com os dados preparados por ``python -m src.common.prepare_data``)::

    python reports/slides/checkpoint1-exemplos/build_checkpoint1_exemplos.py \\
        --input  "C:/Users/<você>/Downloads/checkpoint1.pdf.pdf" \\
        --output "C:/Users/<você>/Downloads/checkpoint1_com_exemplos.pdf" \\
        --png    "C:/Users/<você>/Downloads/checkpoint1_slide_exemplos_fer2013.png"

Fontes (licença OFL, fora do git) em ``build/``: OpenSans-Regular.ttf e OpenSans-Bold.ttf
(github.com/googlefonts/opensans) e IBMPlexSans-Regular.ttf (github.com/IBM/plex).
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.common.data import EMOTIONS, fer2013_arrays, read_clean_csv  # noqa: E402

BUILD = HERE / "build"
FONT_BOLD = BUILD / "OpenSans-Bold.ttf"
FONT_CAPTION = BUILD / "IBMPlexSans-Regular.ttf"
FONT_ARIMO_LIKE = Path("C:/Windows/Fonts/arialbd.ttf")   # a Arimo tem as mesmas medidas da Arial

# Rótulo (ordem oficial do projeto) -> ids escolhidos (linha do fer2013.csv).
EXAMPLES: dict[int, tuple[int, int, int]] = {
    0: (2185, 10155, 20478),    # raiva
    1: (14979, 2389, 17299),    # nojo
    2: (24787, 13561, 5131),    # medo
    3: (24381, 7945, 11942),    # alegria
    4: (7269, 9498, 12803),     # tristeza
    5: (1977, 18241, 16980),    # surpresa
    6: (13808, 9438, 12325),    # neutro
}
TEMPLATE_PAGE = 2   # "2. Contexto: emoções básicas e bases de dados" (layout vertical)
TITLE = "2. Contexto: exemplos da FER-2013"
CAPTION = ("Exemplos escolhidos do conjunto de treino já limpo: rostos de 48×48 pixels em tons de "
           "cinza, coletados da web, com idade, pose, iluminação e enquadramento muito variados.")

DARK = (0x21 / 255, 0x21 / 255, 0x21 / 255)
BLUE = (0x00 / 255, 0x72 / 255, 0xB9 / 255)
FOOTER_RE = re.compile(r"^\d+/\d+$")


def check_examples() -> None:
    """Confere se cada id escolhido é do treino, passou na limpeza sem aviso e tem o rótulo certo."""
    d = read_clean_csv().set_index("id")
    for label, ids in EXAMPLES.items():
        for i in ids:
            row = d.loc[i]
            if row["split"] != "train" or not row["keep"] or row["reason"] != "" or row["label"] != label:
                raise ValueError(f"id {i} não serve como exemplo de {EMOTIONS[label]}: {row.to_dict()}")


def spans(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            yield from line["spans"]


def footer_span(page):
    found = [s for s in spans(page) if FOOTER_RE.match(s["text"].strip()) and s["bbox"][1] > page.rect.height * 0.85]
    if len(found) != 1:
        raise ValueError(f"página {page.number + 1}: esperado um número de rodapé, achei {len(found)}")
    return found[0]


def renumber(page, text: str, force_font: Path | None = None) -> None:
    """Troca só o texto do rodapé, mantendo fonte, tamanho, cor, linha de base e borda direita."""
    s = footer_span(page)
    x0, y0, x1, y1 = s["bbox"]
    page.add_redact_annot(pymupdf.Rect(x0 - 0.5, y0 - 0.5, x1 + 0.5, y1 + 0.5), fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE, graphics=pymupdf.PDF_REDACT_LINE_ART_NONE)
    fontfile = force_font or (FONT_ARIMO_LIKE if "Arimo" in s["font"] else FONT_BOLD)
    font = pymupdf.Font(fontfile=str(fontfile))
    width = font.text_length(text, fontsize=s["size"])
    color = tuple(((s["color"] >> k) & 255) / 255 for k in (16, 8, 0))
    page.insert_text((x1 - width, s["origin"][1]), text, fontsize=s["size"], fontfile=str(fontfile),
                     fontname="F" + fontfile.stem.replace("-", ""), color=color)


def face_png(img: np.ndarray, side: int = 256) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).resize((side, side), Image.LANCZOS).save(buf, format="PNG")
    return buf.getvalue()


def wrap(text: str, font: pymupdf.Font, size: float, width: float) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if font.text_length(trial, fontsize=size) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    return lines + [cur]


def build_gallery(page, src_page, X: np.ndarray) -> None:
    """Limpa a cópia do slide-modelo e desenha título, galeria e nota no padrão do deck."""
    title = next(s for s in spans(src_page) if s["size"] > 15)          # título original (Open Sans Bold)
    for rect in (pymupdf.Rect(10, 8, 405, 45), pymupdf.Rect(10, 45, 405, page.rect.height)):
        page.add_redact_annot(rect, fill=False)
    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE,
                          graphics=pymupdf.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)

    bold = pymupdf.Font(fontfile=str(FONT_BOLD))
    page.insert_text((title["origin"][0], title["origin"][1]), TITLE, fontsize=title["size"],
                     fontfile=str(FONT_BOLD), fontname="FOpenSansBold", color=DARK)

    # Galeria: 7 colunas (ordem oficial) x 3 linhas, dentro da área de conteúdo (à esquerda da faixa azul).
    left, right, tile, row_gap = 24.0, 384.0, 42.0, 5.0
    col_gap = (right - left - 7 * tile) / 6
    label_y, top = 60.0, 66.0
    for c, ids in EXAMPLES.items():
        x = left + c * (tile + col_gap)
        name = EMOTIONS[c]
        size = 10.03                                                   # como "CNN do zero" no slide 5
        page.insert_text((x + tile / 2 - bold.text_length(name, fontsize=size) / 2, label_y), name,
                         fontsize=size, fontfile=str(FONT_BOLD), fontname="FOpenSansBold", color=BLUE)
        for r, i in enumerate(ids):
            y = top + r * (tile + row_gap)
            page.insert_image(pymupdf.Rect(x, y, x + tile, y + tile), stream=face_png(X[i]))

    # Nota em duas linhas, na mesma fonte, tamanho e posição da nota do slide "redes neurais e métodos clássicos".
    plex = pymupdf.Font(fontfile=str(FONT_CAPTION))
    lines = wrap(CAPTION, plex, 8.24, right - 19.8)
    if len(lines) > 2:
        raise ValueError(f"a nota passou de duas linhas: {lines}")
    for line, y in zip(lines, (219.0, 232.5)):
        page.insert_text((19.8, y), line, fontsize=8.24, fontfile=str(FONT_CAPTION), fontname="FIBMPlexSans",
                         color=DARK)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", type=Path, required=True, help="PDF do checkpoint 1 exportado do Canva")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--png", type=Path, help="também salva o slide novo em PNG 1920x1080 (para o Canva)")
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--output não pode ser o próprio arquivo de entrada")
    for f in (FONT_BOLD, FONT_CAPTION, FONT_ARIMO_LIKE):
        if not f.is_file():
            parser.error(f"fonte não encontrada: {f} (veja o docstring)")

    check_examples()
    X = fer2013_arrays()["X"]
    src = pymupdf.open(args.input)
    out = pymupdf.open()
    out.insert_pdf(src)
    new_index = TEMPLATE_PAGE + 1
    out.insert_pdf(src, from_page=TEMPLATE_PAGE, to_page=TEMPLATE_PAGE, start_at=new_index)
    build_gallery(out[new_index], src[TEMPLATE_PAGE], X)

    total = len(out) - 1                              # a capa não é numerada
    for i in range(1, len(out)):
        renumber(out[i], f"{i}/{total}", force_font=FONT_BOLD if i == new_index else None)

    out.set_metadata({**src.metadata, "title": "Checkpoint 1 com exemplos da FER-2013"})
    out.save(args.output, garbage=3, deflate=True)
    print(f"{args.output}: {len(out)} páginas (slide novo na posição {new_index}, numerado {new_index}/{total})")
    if args.png:
        page = out[new_index]
        zoom = 1920 / page.rect.width
        page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).save(args.png)
        print(f"{args.png}: slide novo em PNG")
    return 0


if __name__ == "__main__":
    sys.exit(main())
