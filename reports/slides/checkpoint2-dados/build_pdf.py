"""Gera checkpoint2-dados.pdf sem LaTeX, imitando o tema UFGTeX (Beamer) do checkpoint 1.

Usa os mesmos fundos e logos do tema (lib/) e as mesmas regras de layout do
``beamerthemeUfg.sty``: página de 16:9 (453,6 x 255,4 pt), margens de 1 cm, títulos em \\LARGE
negrito, blocos com barra lateral de 4 pt, tabelas no estilo booktabs com linhas alternadas e
numeração "n/N". O conteúdo é o mesmo de ``checkpoint2-dados.tex``. Quando houver LaTeX,
o .tex continua sendo a fonte oficial.

Uso (da raiz do repositório)::

    python scripts/make_checkpoint2_figures.py           # figuras em reports/figures/
    python reports/slides/checkpoint2-dados/build_pdf.py # gera o PDF nesta pasta
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
LIB = HERE / "lib"
FIGS = REPO / "reports" / "figures"
OUT = HERE / "checkpoint2-dados.pdf"
FONTS = Path("C:/Windows/Fonts")

W, H = 453.602, 255.361
MARGIN = 28.35                     # 1 cm
CONTENT_RIGHT_V = 386.0            # layout vertical: a faixa azul começa à direita disto
CONTENT_TOP = 58.0
CONTENT_BOTTOM = 238.0

UFGBLUE = (0 / 255, 114 / 255, 185 / 255)
DARKGRAY = (33 / 255, 33 / 255, 33 / 255)
LIGHTGRAY_ROW = (244 / 255, 244 / 255, 244 / 255)   # LightGray!10
BLOCK_BG = (0.97, 0.97, 0.97)
SEP_GRAY = (150 / 255, 150 / 255, 150 / 255)
HEX = {"blue": "#0072B9", "gray": "#212121", "orange": "#FF9800", "green": "#5B8D08",
       "darkorange": "#D67A00", "white": "#FFFFFF", "muted": "#6B6B6B"}

ARCHIVE = pymupdf.Archive(str(FONTS))
CSS = """
@font-face {font-family: sui; src: url(segoeui.ttf);}
@font-face {font-family: sui; src: url(segoeuib.ttf); font-weight: bold;}
@font-face {font-family: sui; src: url(segoeuii.ttf); font-style: italic;}
@font-face {font-family: mono; src: url(consola.ttf);}
* {font-family: sui; margin: 0; padding: 0;}
code {font-family: mono; font-size: 92%;}
b {font-weight: bold;}
ul {margin: 0; padding-left: 12px;}
li {margin-bottom: 4px;}
"""

_scratch = pymupdf.open()
_scratch_page = _scratch.new_page(width=W, height=2000)
WARNINGS: list[str] = []


def _html(content: str, size: float, color: str = "gray", align: str = "left", lh: float = 1.25,
          bold: bool = False) -> str:
    weight = "bold" if bold else "normal"
    return (f'<div style="font-size:{size}pt; color:{HEX.get(color, color)}; text-align:{align}; '
            f'line-height:{lh}; font-weight:{weight};">{content}</div>')


def measure(html: str, width: float) -> float:
    """Altura ocupada por um trecho de HTML numa largura dada."""
    rect = pymupdf.Rect(0, 0, width, 1900)
    spare, _ = _scratch_page.insert_htmlbox(rect, html, css=CSS, archive=ARCHIVE, scale_low=1)
    return 1900 - spare


def put(page, rect, html: str, opacity: float = 1.0) -> None:
    spare, scale = page.insert_htmlbox(pymupdf.Rect(rect), html, css=CSS, archive=ARCHIVE,
                                       scale_low=1, opacity=opacity)
    if spare < 0:
        WARNINGS.append(f"página {page.number + 1}: texto não coube em {tuple(round(v) for v in rect)}")


def text(page, x0, y0, x1, html_body: str, size: float, **kw) -> float:
    """Escreve a partir de (x0, y0) e devolve o y final."""
    html = _html(html_body, size, **kw)
    h = measure(html, x1 - x0)
    put(page, (x0, y0, x1, y0 + h + 2), html)
    return y0 + h


# ------------------------------------------------------------------ layouts
def new_page(doc, layout: str, bg=UFGBLUE):
    page = doc.new_page(width=W, height=H)
    page.draw_rect(page.rect, color=None, fill=bg)
    bgfile = {"title": "intro_bg", "vertical": "vertical_bg", "horizontal": "horizontal_bg"}.get(layout)
    if bgfile:
        src = pymupdf.open(LIB / "layouts" / f"{bgfile}.pdf")
        page.show_pdf_page(page.rect, src, 0)
    logo = LIB / "logos" / "INF-compact-white.png"
    lw, lh = pymupdf.open(logo)[0].rect.br
    if layout == "vertical":       # \put(415,195): largura de 10 mm
        w = 28.35
        h = w * lh / lw
        page.insert_image(pymupdf.Rect(415, H - 195 - h, 415 + w, H - 195), filename=logo)
    elif layout == "horizontal":   # \put(15,6): altura de 10 mm x 0,8
        h = 28.35 * 0.8
        w = h * lw / lh
        page.insert_image(pymupdf.Rect(15, H - 6 - h, 15 + w, H - 6), filename=logo)
    elif layout == "title":        # \put(15,15) altura 12 mm; UFG em \put(345,25) altura 15 mm
        h = 34.0
        w = h * lw / lh
        page.insert_image(pymupdf.Rect(15, H - 15 - h, 15 + w, H - 15), filename=logo)
        ufg = pymupdf.open(LIB / "logos" / "ufg_black.pdf")
        uh = 42.5
        uw = uh * ufg[0].rect.width / ufg[0].rect.height
        page.show_pdf_page(pymupdf.Rect(345, H - 25 - uh, 345 + uw, H - 25), ufg, 0)
    return page


def frame_title(page, title: str, white: bool = False, right: float = CONTENT_RIGHT_V) -> None:
    text(page, MARGIN, 17, right, title, 17.28, color="white" if white else "gray", bold=True, lh=1.1)


def page_number(page, n: int, total: int, white: bool = False) -> None:
    color = "white" if white else "gray"
    put(page, (W - 60, H - 15, W - 8, H - 3), _html(f"{n}/{total}", 8, color=color, align="right", bold=True))


def block(page, x0, y0, x1, title: str | None, body: str, kind: str = "block", size: float = 9.5) -> float:
    """Bloco do tema: barra lateral de 4 pt e fundo cinza 0,97. Devolve o y final."""
    bar = {"block": "blue", "alert": "orange", "example": "green"}[kind]
    pad_x, pad_y = 9.0, 5.5
    parts = []
    if title:
        parts.append(f'<div style="color:{HEX[bar]}; font-weight:bold; margin-bottom:2px;">{title}</div>')
    parts.append(f'<div style="color:{HEX["gray"]};">{body}</div>')
    html = _html("".join(parts), size, lh=1.3)
    h = measure(html, x1 - x0 - 4 - 2 * pad_x) + 2 * pad_y
    page.draw_rect((x0, y0, x1, y0 + h), color=None, fill=BLOCK_BG)
    bar_rgb = {"blue": UFGBLUE, "orange": (1, 152 / 255, 0), "green": (91 / 255, 141 / 255, 8 / 255)}[bar]
    page.draw_rect((x0, y0, x0 + 4, y0 + h), color=None, fill=bar_rgb)
    put(page, (x0 + 4 + pad_x, y0 + pad_y, x1 - pad_x, y0 + h), html)
    if y0 + h > CONTENT_BOTTOM + 4:
        WARNINGS.append(f"página {page.number + 1}: bloco termina em y={y0 + h:.0f} (limite {CONTENT_BOTTOM:.0f})")
    return y0 + h


def table(page, x0, y0, widths: list[float], header: list[str], rows: list[list[str]], size: float = 9,
          pad_x: float = 6.0, pad_y: float = 2.6) -> float:
    """Tabela booktabs com linhas alternadas (LightGray!10). Devolve o y final."""
    x1 = x0 + sum(widths)

    def row_height(cells, bold=False):
        return max(measure(_html(c, size, bold=bold, lh=1.2), w - 2 * pad_x) for c, w in zip(cells, widths)) + 2 * pad_y

    y = y0
    page.draw_line((x0, y), (x1, y), color=DARKGRAY, width=0.8)        # toprule
    y += 1.5
    hh = row_height(header, bold=True)
    _cells(page, x0, y, widths, header, size, pad_x, pad_y, hh, bold=True)
    y += hh
    page.draw_line((x0, y), (x1, y), color=DARKGRAY, width=0.5)        # midrule
    y += 1.0
    for i, r in enumerate(rows):
        rh = row_height(r)
        if i % 2 == 1:
            page.draw_rect((x0, y, x1, y + rh), color=None, fill=LIGHTGRAY_ROW)
        _cells(page, x0, y, widths, r, size, pad_x, pad_y, rh)
        y += rh
    y += 1.0
    page.draw_line((x0, y), (x1, y), color=DARKGRAY, width=0.8)        # bottomrule
    return y


def _cells(page, x0, y, widths, cells, size, pad_x, pad_y, h, bold=False):
    x = x0
    for c, w in zip(cells, widths):
        put(page, (x + pad_x, y + pad_y, x + w - pad_x, y + h), _html(c, size, bold=bold, lh=1.2))
        x += w


def figure(page, name: str, cx: float, y0: float, height: float, max_width: float | None = None) -> float:
    src = pymupdf.open(FIGS / f"{name}.pdf")
    r = src[0].rect
    w = height * r.width / r.height
    if max_width and w > max_width:
        w, height = max_width, max_width * r.height / r.width
    page.show_pdf_page(pymupdf.Rect(cx - w / 2, y0, cx + w / 2, y0 + height), src, 0)
    return y0 + height


# ------------------------------------------------------------------ slides
def slide_title(doc):
    p = new_page(doc, "title")
    cx0, cx1 = 40, W - 40
    y = text(p, cx0, 40, cx1, "Preparação e Limpeza dos Dados", 20.7, align="center", bold=True, lh=1.15)
    y = text(p, cx0, y + 4, cx1, "Reconhecimento de Emoções em Imagens Faciais: a base comum das duas frentes",
             9, align="center", bold=True)
    names = ('<table style="margin-left:auto; margin-right:auto;"><tr>'
             '<td style="text-align:right;"><b>Antonio H. Q. V. Lopes</b> (202503993)</td>'
             '<td style="width:22px;"></td>'
             '<td style="text-align:left;"><b>Kauã L. Sandes</b> (202504016)</td></tr><tr>'
             '<td style="text-align:right;"><b>Luccas J. R. Clemente</b> (202504019)</td>'
             '<td></td>'
             '<td style="text-align:left;"><b>Yago L. Coqueiro</b> (202504028)</td></tr></table>')
    y = text(p, cx0, y + 20, cx1, names, 11, align="center", lh=1.35)
    y = text(p, cx0, y + 10, cx1,
             "INF0413 – Processamento de Sinal e Imagem · Prof. Aldo Díaz<br>"
             "Instituto de Informática · Universidade Federal de Goiás<br>"
             "github.com/KauaSandes/seminario-reconhecimento-emocoes", 8, align="center", lh=1.35)
    text(p, cx0, y + 9, cx1, "Prévia do Seminário · Checkpoint 2 · Novembro de 2026", 8.5, align="center",
         bold=True)


def slide_pipeline(doc):
    p = new_page(doc, "vertical")
    frame_title(p, "1. Onde estamos")
    colors = {"one": (33 / 255, 33 / 255, 33 / 255), "nn": UFGBLUE, "classic": (214 / 255, 122 / 255, 0),
              "eval": (91 / 255, 141 / 255, 8 / 255)}
    rows = [
        ("one", "Fase 1<br>Preparar os dados", [("Bases", "FER-2013 e CK+"), ("Limpeza", "duplicatas, rótulos, vazias"),
                                               ("Divisão", "treino, validação, teste")], 1.0),
        ("nn", "Fase 2a", [("CNN do zero", ""), ("Transfer learning", ""), ("Vídeo: CNN + LSTM", "")], 0.33),
        ("classic", "Fase 2b", [("Pré-processamento", ""), ("Descritores", ""), ("SVM e k-NN", "")], 0.33),
        ("eval", "Fase 3", [("Métricas", ""), ("Robustez", ""), ("Interpretação", "")], 0.33),
    ]
    bx0, bw, gap = 112.0, 82.0, 13.0
    y = 60.0
    for key, label, boxes, op in rows:
        bh = 27.0 if key == "one" else 20.0
        lab_hex = {"one": "#212121", "nn": "#0072B9", "classic": "#D67A00", "eval": "#5B8D08"}[key]
        put(p, (MARGIN - 6, y + (bh - (24 if key == "one" else 11)) / 2, bx0 - 8, y + bh + 4),
            _html(label, 8, color=lab_hex, align="right", bold=True, lh=1.1), opacity=1.0 if key == "one" else 0.5)
        for i, (t, sub) in enumerate(boxes):
            x = bx0 + i * (bw + gap)
            p.draw_rect((x, y, x + bw, y + bh), color=None, fill=colors[key], fill_opacity=op, radius=0.12)
            body = f"<b>{t}</b>" + (f"<br>{sub}" if sub else "")
            th = measure(_html(body, 7.5, color="white", align="center", lh=1.15), bw - 4)
            put(p, (x + 2, y + (bh - th) / 2, x + bw - 2, y + bh), _html(body, 7.5, color="white", align="center", lh=1.15),
                opacity=1.0 if key == "one" else 0.75)
            if key == "one" and i < 2:
                ax = x + bw
                p.draw_line((ax + 2, y + bh / 2), (ax + gap - 3, y + bh / 2), color=DARKGRAY, width=1.0)
                p.draw_polyline([(ax + gap - 6, y + bh / 2 - 2.5), (ax + gap - 2.5, y + bh / 2),
                                 (ax + gap - 6, y + bh / 2 + 2.5)], color=DARKGRAY, fill=DARKGRAY, width=0.5)
        y += bh + (11 if key == "one" else 6)
    block(p, MARGIN, y + 10, CONTENT_RIGHT_V, "Esta rodada: a Fase 1, que as duas frentes compartilham",
          "Mesmos dados, mesma limpeza, mesma divisão e mesmas métricas para redes e clássicos.")


def slide_sources(doc):
    p = new_page(doc, "vertical")
    frame_title(p, "2. Obtenção dos dados")
    y = table(p, MARGIN, CONTENT_TOP, [58, 150, 150], ["", "FER-2013", "CK+"], [
        ["Fonte", "Competição oficial do Kaggle", "Pedido aos autores (Lucey et al., 2010)"],
        ["Conteúdo", "35.887 rostos 48×48, 7 classes", "593 sequências, 327 rotuladas"],
        ["Divisão", "oficial: 28.709 / 3.589 / 3.589", "10 partes, agrupadas por pessoa"],
        ["Situação", '<b style="color:#5B8D08;">baixada e validada</b> (SHA-256)',
         '<b style="color:#D67A00;">aguardando acesso</b>; código pronto e testado'],
    ])
    block(p, MARGIN, y + 16, CONTENT_RIGHT_V, "Por que não usar as cópias em pastas de imagens",
          "Versões como a <code>msambare/fer2013</code> perderam a separação PublicTest/PrivateTest: "
          "a validação e o teste oficiais deixariam de existir.", kind="alert")


def slide_common(doc):
    p = new_page(doc, "vertical")
    frame_title(p, "3. A base comum: <code>src/common</code>")
    y = table(p, MARGIN, CONTENT_TOP, [64, 140, 154], ["Módulo", "O que oferece", "O que garante"], [
        ["<code>data.py</code>", "<code>load_fer2013</code>, <code>load_ckplus</code>", "os mesmos dados para todos"],
        ["<code>clean.py</code>", "limpeza e <code>class_weights</code>", "a mesma limpeza para todos"],
        ["<code>degrade.py</code>", "ruído, desfoque e resolução", "as mesmas degradações"],
        ["<code>metrics.py</code>", "<code>evaluate</code>, <code>save_results</code>", "um só formato de resultado"],
    ])
    body = ('<table style="width:100%;"><tr><td><code>python scripts/download_data.py</code></td>'
            '<td style="text-align:right;">baixa, extrai e valida</td></tr>'
            '<tr><td><code>python -m src.common.prepare_data</code></td>'
            '<td style="text-align:right;">limpa, divide e relata</td></tr></table>')
    block(p, MARGIN, y + 14, CONTENT_RIGHT_V, "Dois comandos preparam tudo", body, size=9)


def slide_rules(doc):
    p = new_page(doc, "vertical")
    frame_title(p, "4. Limpeza: o que procuramos")
    y = table(p, MARGIN, CONTENT_TOP, [184, 82, 92], ["Problema", "Treino", "Validação e teste"], [
        ["Duplicata exata (mesmos pixels)", "fica uma cópia", "só sinaliza"],
        ["Cópia de imagem da validação/teste", "sai (vazamento)", "intactas"],
        ["Cópias com rótulos diferentes", "saem", "só sinaliza"],
        ["Imagem vazia (desvio padrão &lt; 5)", "sai", "só sinaliza"],
        ["Sem rosto detectado (Viola-Jones)", "só sinaliza", "só sinaliza"],
    ])
    block(p, MARGIN, y + 12, CONTENT_RIGHT_V, "Nada é apagado",
          "O arquivo versionado <code>splits/fer2013_clean.csv</code> diz o que entra. Validação e teste "
          "ficam intactos, comparáveis com a literatura.")


def slide_result(doc):
    p = new_page(doc, "blank")
    frame_title(p, "5. Limpeza: resultado", white=True, right=W - MARGIN)
    stats = [("26.911", "imagens de treino<br>(eram 28.709)"), ("703", "cópias de imagens da<br>validação e do teste"),
             ("0", "alterações na<br>validação e no teste")]
    colw = (W - 2 * MARGIN) / 3
    top = 92.0
    for i, (num, lab) in enumerate(stats):
        x0 = MARGIN + i * colw
        put(p, (x0, top, x0 + colw, top + 50), _html(num, 40, color="white", align="center", bold=True, lh=1.0))
        put(p, (x0 + 6, top + 54, x0 + colw - 6, top + 85), _html(lab, 9, color="white", align="center", lh=1.25))
        if i:
            p.draw_line((x0, top + 4), (x0, top + 4 + 56.7), color=SEP_GRAY, width=1.0)
    text(p, MARGIN, 196, W - MARGIN,
         "1.798 imagens saíram do treino (6,3%). Sem a limpeza, 703 imagens de avaliação já teriam sido "
         "vistas no treino, e a acurácia medida ficaria otimista.", 9, color="white")


def slide_figure(doc, title: str, fig: str, fig_h: float, block_title: str | None, block_body: str):
    p = new_page(doc, "vertical")
    frame_title(p, title)
    y = figure(p, fig, (MARGIN + CONTENT_RIGHT_V) / 2, CONTENT_TOP - 2, fig_h, max_width=CONTENT_RIGHT_V - MARGIN)
    block(p, MARGIN, y + 8, CONTENT_RIGHT_V, block_title, block_body)


def slide_degradations(doc):
    p = new_page(doc, "vertical")
    frame_title(p, "9. Robustez: degradações prontas")
    y = figure(p, "degradations_example_face", 118, CONTENT_TOP - 4, 168)
    text(p, MARGIN, y + 1, 210, "Rosto de exemplo: imagem <i>astronaut</i> do scikit-image (domínio público).",
         6, color="muted", align="center")
    items = ("<ul>"
             "<li>Aplicadas <b>só no teste</b></li>"
             "<li>Mesmos níveis e mesma semente nas duas frentes</li>"
             "<li>Ruído gaussiano: tema do <b>le03</b></li>"
             "<li>Resolução: média por área (antialias) e volta a 48 px; amostragem e Nyquist do <b>le04</b></li>"
             "</ul>")
    y2 = text(p, 222, CONTENT_TOP + 4, CONTENT_RIGHT_V, items, 9.5, lh=1.3)
    block(p, 222, y2 + 12, CONTENT_RIGHT_V, None,
          "Saída: curvas de acurácia por nível, redes contra clássicos.", size=9)


def slide_quality(doc):
    p = new_page(doc, "vertical")
    frame_title(p, "10. Qualidade e reprodutibilidade")
    y = table(p, MARGIN, CONTENT_TOP, [128, 230], ["Garantia", "Como"], [
        ["Código testado", "87 testes automáticos (pytest)"],
        ["Mesmo arquivo em toda máquina", "SHA-256 do <code>fer2013.csv</code> conferido pelo script"],
        ["Resultados comparáveis", "seed 42 e um único JSON de métricas"],
        ["Sem dados no GitHub", "cada um baixa da fonte oficial"],
    ])
    block(p, MARGIN, y + 14, CONTENT_RIGHT_V, "Primeira linha de base: classe majoritária",
          "Prever sempre “alegria” acerta <b>24,9%</b> da validação (F1 macro 0,06). "
          "É o piso para todos os métodos.")


def slide_next(doc):
    p = new_page(doc, "vertical")
    frame_title(p, "11. Próximos passos")
    items = ("<ul>"
             "<li><b>Frente B:</b> pré-processamento (Viola-Jones, equalização, gaussiano) e os descritores "
             "Gabor, HOG, LBP e DCT</li>"
             "<li><b>Frente A:</b> ligar o pipeline à base comum e treinar a CNN do zero e a ResNet-18</li>"
             "<li><b>CK+:</b> aguardando a resposta dos autores; o código já está pronto e testado</li>"
             "<li>Revisão cruzada da base comum e merge na <code>main</code></li>"
             "</ul>")
    y = text(p, MARGIN, CONTENT_TOP + 4, CONTENT_RIGHT_V, items, 10.5, lh=1.3)
    block(p, MARGIN, y + 14, CONTENT_RIGHT_V, "Situação",
          "A base comum está pronta: as duas frentes já podem treinar sobre os mesmos dados.")


def slide_thanks(doc):
    p = new_page(doc, "blank")
    text(p, 0, 74, W, "Obrigado", 24.88, color="white", align="center", bold=True)
    text(p, 0, 116, W, "Dúvidas e sugestões", 11, color="white", align="center", bold=True)
    logo = LIB / "logos" / "INF-full-white.png"
    lw, lh = pymupdf.open(logo)[0].rect.br
    h = 85.0
    w = h * lw / lh
    p.insert_image(pymupdf.Rect(W / 2 - w / 2, 150, W / 2 + w / 2, 150 + h), filename=logo)


def main() -> int:
    doc = pymupdf.open()
    slide_title(doc)
    slide_pipeline(doc)
    slide_sources(doc)
    slide_common(doc)
    slide_rules(doc)
    slide_result(doc)
    slide_figure(doc, "6. O que saiu do treino", "fer2013_cleaning_removals_by_reason", 128, None,
                 "As duplicatas se concentram em <b>surpresa</b>: 711 das 1.798 imagens removidas.")
    slide_figure(doc, "7. Imagens vazias e o limiar", "fer2013_std_histogram", 118,
                 "O desvio padrão dos pixels mede o contraste: 0 é uma imagem lisa",
                 "Há um vão natural entre 4,1 e 9,1. O limiar 5 pega as 14 imagens vazias (12 no treino) "
                 "e preserva todos os rostos. Conferido visualmente antes de ser fixado.")
    slide_figure(doc, "8. Distribuição de classes", "fer2013_class_distribution", 120,
                 "O desbalanceamento continua: nojo tem 1,3% do treino",
                 "<code>class_weights(y)</code> (inverso da frequência, igual ao <i>balanced</i> do scikit-learn) "
                 "dá peso 10,98 ao nojo e 0,55 à alegria, nas duas frentes.")
    slide_degradations(doc)
    slide_quality(doc)
    slide_next(doc)
    slide_thanks(doc)

    total = len(doc) - 2  # capa e encerramento sem número, como no checkpoint 1
    for i in range(1, len(doc) - 1):
        page_number(doc[i], i, total, white=(i == 5))
    doc.set_metadata({"title": "Checkpoint 2: preparação e limpeza dos dados",
                      "author": "Antonio H. Q. V. Lopes, Kauã L. Sandes, Luccas J. R. Clemente, Yago L. Coqueiro",
                      "subject": "INF0413 – Reconhecimento de emoções em imagens faciais"})
    doc.save(OUT, garbage=3, deflate=True)
    print(f"{OUT} ({len(doc)} páginas)")
    for w in WARNINGS:
        print("AVISO:", w)
    return 1 if WARNINGS else 0


if __name__ == "__main__":
    sys.exit(main())
