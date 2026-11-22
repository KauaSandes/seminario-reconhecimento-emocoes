// Gera checkpoint2-dados.pptx com a identidade visual do checkpoint 1 (tema UFGTeX).
//
// Uso (da raiz do repositório):
//   python reports/slides/checkpoint2-dados/pptx_assets.py   # fundos, logos e dados dos gráficos
//   npm install pptxgenjs                                     # uma vez (fora do repositório, se preferir)
//   node reports/slides/checkpoint2-dados/build_pptx.js
//
// Mesmo conteúdo do PDF (build_pdf.py) e do .tex. Gráficos nativos do PowerPoint (editáveis);
// a grade das degradações entra como imagem. Coordenadas = as do PDF (453,6 x 255,4 pt) escaladas
// para o slide 16:9 do PowerPoint (10 x 5,625 pol).
"use strict";
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const BUILD = path.join(HERE, "pptx_build");
const OUT = path.join(HERE, "checkpoint2-dados.pptx");
const DATA = JSON.parse(fs.readFileSync(path.join(BUILD, "chart_data.json"), "utf8"));
const asset = (f) => path.join(BUILD, f);

// Tema: as cores do beamerthemeUfg.sty
const THEME = {
	name: "UFG INF0413",
	headFontFace: "Calibri",
	bodyFontFace: "Calibri",
	colors: {
		dk1: "212121", lt1: "FFFFFF", dk2: "005CA1", lt2: "F4F4F4",
		accent1: "0072B9", accent2: "C76E00", accent3: "5B8D08", accent4: "FF9800",
		accent5: "005CA1", accent6: "969696", hlink: "0072B9", folHlink: "8E24AA",
	},
};
const HEX = { blue: "0072B9", orange: "C76E00", gray: "212121", muted: "6B6B6B", grid: "E3E3E3", row: "F4F4F4" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5,625 pol
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Checkpoint 2: preparação e limpeza dos dados";
pres.subject = "INF0413 – Reconhecimento de emoções em imagens faciais";
pres.author = "Antonio H. Q. V. Lopes, Kauã L. Sandes, Luccas J. R. Clemente, Yago L. Coqueiro";
const C = pres.SchemeColor;

const W = 10, H = 5.625;
const M = 0.625;                 // margem de 1 cm do tema, escalada
const RIGHT = 8.51;              // a faixa azul do layout vertical começa à direita disto
const CW = RIGHT - M;
const TOTAL = 11;

// ------------------------------------------------------------------ layouts
const INF_ASPECT = 171 / 185;
pres.defineSlideMaster({
	title: "CAPA",
	background: { path: asset("intro_bg.png") },
	objects: [
		{ image: { path: asset("INF-compact-white.png"), x: 0.33, y: H - 0.33 - 0.75, w: 0.75 / INF_ASPECT, h: 0.75 } },
		{ image: { path: asset("ufg_black.png"), x: 7.6, y: 4.14, w: 0.937 * 1131 / 598, h: 0.937 } },
	],
});
pres.defineSlideMaster({
	title: "CONTEUDO",
	background: { path: asset("vertical_bg.png") },
	objects: [
		{ image: { path: asset("INF-compact-white.png"), x: 9.15, y: 1.07, w: 0.625, h: 0.625 * INF_ASPECT } },
		{ placeholder: { options: { name: "title", type: "title", x: M, y: 0.27, w: CW, h: 0.62, fontSize: 27,
			bold: true, color: C.text1, align: "left", valign: "middle", margin: 0 }, text: "" } },
	],
});
pres.defineSlideMaster({
	title: "DESTAQUE_AZUL",
	background: { color: C.accent1 },
	objects: [
		{ placeholder: { options: { name: "title", type: "title", x: M, y: 0.27, w: W - 2 * M, h: 0.62, fontSize: 27,
			bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } },
	],
});
pres.defineSlideMaster({ title: "ENCERRAMENTO", background: { color: C.accent1 } });

// ------------------------------------------------------------------ helpers
function pageNumber(slide, n, white = false) {
	slide.addText(`${n}/${TOTAL}`, { x: W - 1.3, y: H - 0.36, w: 1.12, h: 0.28, fontSize: 12, bold: true,
		align: "right", color: white ? C.background1 : C.text1, margin: 0, isTextBox: true, objectName: "Número" });
}

// Bloco do tema: fundo cinza e barra lateral de 4 pt (identidade do UFGTeX).
function block(slide, { x = M, y, w = CW, h, title, body, kind = "block", size = 15 }) {
	const bar = { block: C.accent1, alert: C.accent4, example: C.accent3 }[kind];
	slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { type: "none" },
		objectName: "Bloco fundo" });
	slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.088, h, fill: { color: bar }, line: { type: "none" },
		objectName: "Bloco barra" });
	const runs = [];
	if (title) runs.push({ text: title, options: { bold: true, color: bar, breakLine: true } });
	for (const r of body) runs.push(r);
	slide.addText(runs, { x: x + 0.088 + 0.2, y, w: w - 0.088 - 0.4, h, fontSize: size, color: C.text1,
		valign: "middle", margin: 0, paraSpaceAfter: 2, isTextBox: true, objectName: title || "Bloco" });
}

const code = (t, extra = {}) => ({ text: t, options: { fontFace: "Consolas", ...extra } });
const b = (t, extra = {}) => ({ text: t, options: { bold: true, ...extra } });
const t = (s, extra = {}) => ({ text: s, options: { ...extra } });

// Tabela no estilo booktabs: regra grossa no topo e no fim, fina sob o cabeçalho, linhas alternadas.
function table(slide, { x = M, y, colW, header, rows, fontSize = 14, rowH }) {
	const none = { type: "none" };
	const thick = { type: "solid", pt: 1.25, color: HEX.gray };
	const thin = { type: "solid", pt: 0.75, color: HEX.gray };
	const cell = (content, opts) => ({
		text: typeof content === "string" ? content : content,
		options: { margin: [3, 6, 3, 6], valign: "middle", fontSize, color: C.text1, ...opts },
	});
	const out = [header.map((hd) => cell(hd, { bold: true, border: [thick, none, thin, none] }))];
	rows.forEach((r, i) => {
		const last = i === rows.length - 1;
		out.push(r.map((c) => cell(c, {
			fill: { color: i % 2 === 1 ? C.background2 : C.background1 },
			border: [none, none, last ? thick : none, none],
		})));
	});
	slide.addTable(out, { x, y, w: colW.reduce((a, v) => a + v, 0), colW, rowH, autoPage: false, objectName: "Tabela" });
}

const axisFont = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12,
	valAxisLabelFontSize: 12, catAxisLabelColor: HEX.gray, valAxisLabelColor: HEX.muted };

// ------------------------------------------------------------------ slides
pres.addSection({ title: "Abertura" });

// Capa
{
	const s = pres.addSlide({ masterName: "CAPA", sectionTitle: "Abertura" });
	s.addText("Preparação e Limpeza dos Dados", { x: 0.9, y: 0.82, w: 8.2, h: 0.75, fontSize: 32, bold: true,
		align: "center", color: C.text1, margin: 0, isTextBox: true, objectName: "Título" });
	s.addText("Reconhecimento de Emoções em Imagens Faciais: a base comum das duas frentes", { x: 0.9, y: 1.58,
		w: 8.2, h: 0.35, fontSize: 14, bold: true, align: "center", color: C.text1, margin: 0, isTextBox: true,
		objectName: "Subtítulo" });
	const names = [["Antonio H. Q. V. Lopes", "(202503993)", "Kauã L. Sandes", "(202504016)"],
		["Luccas J. R. Clemente", "(202504019)", "Yago L. Coqueiro", "(202504028)"]];
	names.forEach((row, i) => {
		const y = 2.2 + i * 0.4;
		s.addText([b(row[0]), t(" " + row[1])], { x: 0.6, y, w: 4.25, h: 0.36, fontSize: 17, align: "right",
			color: C.text1, margin: 0, isTextBox: true, objectName: `Autor ${2 * i + 1}` });
		s.addText([b(row[2]), t(" " + row[3])], { x: 5.2, y, w: 4.25, h: 0.36, fontSize: 17, align: "left",
			color: C.text1, margin: 0, isTextBox: true, objectName: `Autor ${2 * i + 2}` });
	});
	s.addText([
		t("INF0413 – Processamento de Sinal e Imagem · Prof. Aldo Díaz", { breakLine: true }),
		t("Instituto de Informática · Universidade Federal de Goiás", { breakLine: true }),
		t("github.com/KauaSandes/seminario-reconhecimento-emocoes"),
	], { x: 0.9, y: 3.18, w: 8.2, h: 0.72, fontSize: 12.5, align: "center", color: C.text1, margin: 0,
		isTextBox: true, objectName: "Instituição" });
	s.addText("Prévia do Seminário · Checkpoint 2 · Novembro de 2026", { x: 2.0, y: 4.0, w: 5.4, h: 0.32,
		fontSize: 13.5, bold: true, align: "center", color: C.text1, margin: 0, isTextBox: true, objectName: "Data" });
	s.addNotes("Checkpoint 2: nesta rodada a frente B (Yago e Luccas) preparou os dados e a base comum que as duas frentes usam.");
}

pres.addSection({ title: "Dados e limpeza" });

// 1. Onde estamos
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("1. Onde estamos", { placeholder: "title" });
	const rows = [
		{ label: "Fase 1\nPreparar os dados", color: C.text1, hex: HEX.gray, active: true,
			boxes: [["Bases", "FER-2013 e CK+"], ["Limpeza", "duplicatas, rótulos, vazias"], ["Divisão", "treino, validação, teste"]] },
		{ label: "Fase 2a", color: C.accent1, boxes: [["CNN do zero"], ["Transfer learning"], ["Vídeo: CNN + LSTM"]] },
		{ label: "Fase 2b", color: C.accent2, boxes: [["Pré-processamento"], ["Descritores"], ["SVM e k-NN"]] },
		{ label: "Fase 3", color: C.accent3, boxes: [["Métricas"], ["Robustez"], ["Interpretação"]] },
	];
	const bx0 = 2.47, bw = 1.81, gap = 0.29;
	let y = 1.3;
	rows.forEach((r) => {
		const bh = r.active ? 0.6 : 0.44;
		s.addText(r.label, { x: M - 0.15, y, w: bx0 - M - 0.05, h: bh, fontSize: 12.5, bold: true, align: "right",
			valign: "middle", color: r.color, margin: 0, isTextBox: true, objectName: r.label.split("\n")[0] });
		r.boxes.forEach(([title, sub], i) => {
			const x = bx0 + i * (bw + gap);
			const runs = [b(title, sub ? { breakLine: true } : {})];
			if (sub) runs.push(t(sub, { fontSize: 11 }));
			s.addText(runs, {
				shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x, y, w: bw, h: bh,
				fill: r.active ? { color: C.text1 } : { color: r.color, transparency: 82 },
				color: r.active ? C.background1 : r.color, fontSize: 12, align: "center", valign: "middle",
				margin: 2, line: { type: "none" }, objectName: title,
			});
			if (r.active && i < 2) {
				s.addShape(pres.shapes.LINE, { x: x + bw + 0.04, y: y + bh / 2, w: gap - 0.08, h: 0,
					line: { color: C.text1, width: 1.25, endArrowType: "triangle" }, objectName: "Seta" });
			}
		});
		y += bh + (r.active ? 0.24 : 0.13);
	});
	block(s, { y: y + 0.18, h: 0.92, title: "Esta rodada: a Fase 1, que as duas frentes compartilham",
		body: [t("Mesmos dados, mesma limpeza, mesma divisão e mesmas métricas para redes e clássicos.")] });
	pageNumber(s, 1);
	s.addNotes("A metodologia do checkpoint 1 tem quatro fases. Esta rodada concluiu a Fase 1, que é comum: as redes (2a) e os métodos clássicos (2b) vão usar exatamente os mesmos dados.");
}

// 2. Obtenção dos dados
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("2. Obtenção dos dados", { placeholder: "title" });
	table(s, { y: 1.27, colW: [1.28, 3.3, 3.3], header: ["", "FER-2013", "CK+"], rows: [
		["Fonte", "Competição oficial do Kaggle", "Pedido aos autores (Lucey et al., 2010)"],
		["Conteúdo", "35.887 rostos 48×48, 7 classes", "593 sequências, 327 rotuladas"],
		["Divisão", "oficial: 28.709 / 3.589 / 3.589", "10 partes, agrupadas por pessoa"],
		["Situação", [b("baixada e validada", { color: C.accent3 }), t(" (SHA-256)")],
			[b("aguardando acesso", { color: C.accent2 }), t("; código pronto e testado")]],
	] });
	block(s, { y: 3.62, h: 1.05, kind: "alert", title: "Por que não usar as cópias em pastas de imagens",
		body: [t("Versões como a "), code("msambare/fer2013"), t(" perderam a separação PublicTest/PrivateTest: a validação e o teste oficiais deixariam de existir.")] });
	pageNumber(s, 2);
	s.addNotes("A FER-2013 veio da competição oficial do Kaggle, a única fonte que mantém a divisão oficial. O SHA-256 garante que todas as máquinas têm o mesmo arquivo. O CK+ depende de autorização dos autores; o código já está pronto e testado com uma pasta falsa.");
}

// 3. Base comum
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText([t("3. A base comum: "), code("src/common", { bold: false })], { placeholder: "title" });
	table(s, { y: 1.27, colW: [1.41, 3.09, 3.39], header: ["Módulo", "O que oferece", "O que garante"], rows: [
		[[code("data.py")], [code("load_fer2013"), t(", "), code("load_ckplus")], "os mesmos dados para todos"],
		[[code("clean.py")], [t("limpeza e "), code("class_weights")], "a mesma limpeza para todos"],
		[[code("degrade.py")], "ruído, desfoque e resolução", "as mesmas degradações"],
		[[code("metrics.py")], [code("evaluate"), t(", "), code("save_results")], "um só formato de resultado"],
	] });
	block(s, { y: 3.4, h: 1.12, title: "Dois comandos preparam tudo", size: 14, body: [
		code("python scripts/download_data.py"), t("   baixa, extrai e valida", { color: C.text1, breakLine: true }),
		code("python -m src.common.prepare_data"), t("   limpa, divide e gera o relatório"),
	] });
	pageNumber(s, 3);
	s.addNotes("Tudo que as duas frentes usam sai de src/common: carregamento, limpeza, degradações e métricas. Assim a comparação entre redes e clássicos é justa.");
}

// 4. Regras da limpeza
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("4. Limpeza: o que procuramos", { placeholder: "title" });
	table(s, { y: 1.27, colW: [4.06, 1.81, 2.02], header: ["Problema", "Treino", "Validação e teste"], rows: [
		["Duplicata exata (mesmos pixels)", "fica uma cópia", "só sinaliza"],
		["Cópia de imagem da validação/teste", "sai (vazamento)", "intactas"],
		["Cópias com rótulos diferentes", "saem", "só sinaliza"],
		["Imagem vazia (desvio padrão < 5)", "sai", "só sinaliza"],
		["Sem rosto detectado (Viola-Jones)", "só sinaliza", "só sinaliza"],
	] });
	block(s, { y: 3.72, h: 1.0, title: "Nada é apagado", body: [
		t("O arquivo versionado "), code("splits/fer2013_clean.csv"),
		t(" diz o que entra. Validação e teste ficam intactos, comparáveis com a literatura."),
	] });
	pageNumber(s, 4);
	s.addNotes("Só o treino perde imagens. Na validação e no teste os problemas são apenas sinalizados, para os números continuarem comparáveis com os artigos que usam a divisão oficial.");
}

// 5. Resultado (slide azul com números)
{
	const s = pres.addSlide({ masterName: "DESTAQUE_AZUL", sectionTitle: "Dados e limpeza" });
	s.addText("5. Limpeza: resultado", { placeholder: "title" });
	const stats = [["26.911", "imagens de treino\n(eram 28.709)"], ["703", "cópias de imagens da\nvalidação e do teste"],
		["0", "alterações na\nvalidação e no teste"]];
	const colw = (W - 2 * M) / 3;
	stats.forEach(([num, lab], i) => {
		const x = M + i * colw;
		s.addText(num, { x, y: 1.95, w: colw, h: 1.05, fontSize: 60, bold: true, align: "center", valign: "middle",
			color: C.background1, margin: 0, isTextBox: true, objectName: `Número ${i + 1}` });
		s.addText(lab, { x: x + 0.1, y: 3.08, w: colw - 0.2, h: 0.62, fontSize: 14, align: "center", valign: "top",
			color: C.background1, margin: 0, isTextBox: true, objectName: `Rótulo ${i + 1}` });
		if (i) s.addShape(pres.shapes.LINE, { x, y: 2.05, w: 0, h: 1.25, line: { color: "969696", width: 1 },
			objectName: "Separador" });
	});
	s.addText("1.798 imagens saíram do treino (6,3%). Sem a limpeza, 703 imagens de avaliação já teriam sido vistas no treino, e a acurácia medida ficaria otimista.",
		{ x: M, y: 4.2, w: W - 2 * M, h: 0.62, fontSize: 14, color: C.background1, margin: 0, isTextBox: true,
			objectName: "Nota" });
	pageNumber(s, 5, true);
	s.addNotes("O treino caiu de 28.709 para 26.911 imagens. O ponto mais importante: 703 imagens do treino eram cópias de imagens da validação ou do teste. Sem tirá-las, o modelo seria avaliado em imagens que já viu.");
}

// 6. O que saiu do treino (gráfico nativo)
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("6. O que saiu do treino", { placeholder: "title" });
	s.addChart(pres.charts.BAR, [{ name: "Imagens removidas", labels: DATA.removals.labels, values: DATA.removals.values }], {
		x: M, y: 1.15, w: CW, h: 2.95, barDir: "bar", chartColors: [HEX.blue], barGapWidthPct: 55,
		showValue: true, dataLabelPosition: "outEnd", dataLabelFontSize: 13, dataLabelFontBold: true,
		dataLabelColor: HEX.gray, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: "#,##0",
		valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false,
		showTitle: true, title: `${DATA.removals.total.toLocaleString("pt-BR")} imagens removidas (uma imagem pode ter mais de um motivo)`,
		titleFontSize: 12, titleColor: HEX.muted, titleFontFace: "+mn-lt", ...axisFont, objectName: "Gráfico de remoções",
	});
	block(s, { y: 4.3, h: 0.6, body: [t("As duplicatas se concentram em "), b("surpresa"), t(": 711 das 1.798 imagens removidas.")] });
	pageNumber(s, 6);
	s.addNotes("A maior parte das remoções são duplicatas dentro do treino (1.021) e cópias de imagens da validação ou do teste (703). Uma imagem pode ter mais de um motivo, por isso as barras somam mais que 1.798.");
}

// 7. Imagens vazias e o limiar (histograma nativo, escala log)
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("7. Imagens vazias e o limiar", { placeholder: "title" });
	const hd = DATA.histogram;
	s.addChart(pres.charts.BAR, [
		{ name: `Vazias (desvio < ${hd.threshold}): ${hd.n_low} imagens`, labels: hd.labels, values: hd.low },
		{ name: "Demais imagens", labels: hd.labels, values: hd.high },
	], {
		x: M, y: 1.12, w: CW, h: 2.85, barDir: "col", barGrouping: "clustered", barOverlapPct: 100, barGapWidthPct: 15,
		chartColors: [HEX.orange, HEX.blue], valAxisLogScaleBase: 10, valAxisMinVal: 0.1, valAxisCrossesAt: 0.1,
		valAxisLabelFormatCode: "General", valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
		catAxisLabelFrequency: 10, showLegend: true, legendPos: "t", legendFontSize: 12, legendFontFace: "+mn-lt",
		legendColor: HEX.gray, showValAxisTitle: true, valAxisTitle: "imagens (escala log)", valAxisTitleFontSize: 12,
		valAxisTitleColor: HEX.muted, showCatAxisTitle: true, catAxisTitle: "desvio padrão dos pixels de cada imagem",
		catAxisTitleFontSize: 12, catAxisTitleColor: HEX.muted, ...axisFont, objectName: "Histograma do desvio padrão",
	});
	block(s, { y: 4.08, h: 1.0, title: "O desvio padrão dos pixels mede o contraste: 0 é uma imagem lisa", size: 14, body: [
		t("Há um vão natural entre 4,1 e 9,1. O limiar 5 pega as 14 imagens vazias (12 no treino) e preserva todos os rostos. Conferido visualmente antes de ser fixado."),
	] });
	pageNumber(s, 7);
	s.addNotes("O desvio padrão dos 2.304 pixels mede quanto a imagem varia. Uma imagem toda preta tem desvio 0. Rostos ficam entre 15 e 115. O limiar 5 separa as 14 imagens vazias, que conferimos uma a uma na grade.");
}

// 8. Distribuição de classes (gráfico nativo)
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("8. Distribuição de classes", { placeholder: "title" });
	const dd = DATA.distribution;
	const fmt = (n) => n.toLocaleString("pt-BR");
	s.addChart(pres.charts.BAR, [
		{ name: `Treino antes (${fmt(dd.n_before)})`, labels: dd.labels, values: dd.before },
		{ name: `Treino depois (${fmt(dd.n_after)})`, labels: dd.labels, values: dd.after },
	], {
		x: M, y: 1.12, w: CW, h: 2.85, barDir: "col", barGrouping: "clustered", barGapWidthPct: 60,
		chartColors: [HEX.orange, HEX.blue], valAxisLabelFormatCode: "0", valGridLine: { color: HEX.grid, size: 0.75 },
		catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 12, legendFontFace: "+mn-lt",
		legendColor: HEX.gray, showValAxisTitle: true, valAxisTitle: "% do treino", valAxisTitleFontSize: 12,
		valAxisTitleColor: HEX.muted, ...axisFont, objectName: "Distribuição de classes",
	});
	block(s, { y: 4.08, h: 1.0, title: "O desbalanceamento continua: nojo tem 1,3% do treino", size: 14, body: [
		code("class_weights(y)"), t(" (inverso da frequência, igual ao "), t("balanced", { italic: true }),
		t(" do scikit-learn) dá peso 10,98 ao nojo e 0,55 à alegria, nas duas frentes."),
	] });
	pageNumber(s, 8);
	s.addNotes("A limpeza quase não mudou as proporções; surpresa caiu de 11,0% para 9,1% porque concentrava duplicatas. Nojo continua com 1,3%, então as duas frentes usam os mesmos pesos de classe.");
}

// 9. Degradações (imagem: é uma grade de fotos, não um gráfico)
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("9. Robustez: degradações prontas", { placeholder: "title" });
	const ih = 3.62, iw = ih * 757 / 605;
	s.addImage({ path: asset("degradations_example_face.png"), x: M - 0.1, y: 1.17, w: iw, h: ih,
		altText: "Um rosto de exemplo sob ruído gaussiano (sigma 5, 10, 20), desfoque gaussiano (kernel 3, 5, 7) e redução de resolução (24 e 12 px, de volta a 48).",
		objectName: "Exemplo de degradações" });
	s.addText("Rosto de exemplo: imagem astronaut do scikit-image (domínio público).", { x: M - 0.1, y: 4.84, w: iw,
		h: 0.25, fontSize: 10, align: "center", color: HEX.muted, margin: 0, isTextBox: true, objectName: "Legenda" });
	s.addText([
		t("Aplicadas ", { bullet: true }), b("só no teste", { breakLine: true }),
		t("Mesmos níveis e mesma semente nas duas frentes", { bullet: true, breakLine: true }),
		t("Ruído gaussiano: tema do ", { bullet: true }), b("le03", { breakLine: true }),
		t("Resolução: média por área (antialias) e volta a 48 px; amostragem e Nyquist do ", { bullet: true }), b("le04"),
	], { x: 5.0, y: 1.3, w: RIGHT - 5.0, h: 2.45, fontSize: 15, color: C.text1, paraSpaceAfter: 6, valign: "top",
		margin: 0, isTextBox: true, objectName: "Pontos" });
	block(s, { x: 5.0, y: 3.95, w: RIGHT - 5.0, h: 0.85, size: 14,
		body: [t("Saída: curvas de acurácia por nível, redes contra clássicos.")] });
	pageNumber(s, 9);
	s.addNotes("As três degradações do contrato já estão implementadas em src/common/degrade.py, com os níveis definidos no checkpoint 1. A redução de resolução usa média por área, que funciona como filtro antialias antes de amostrar.");
}

// 10. Qualidade
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Dados e limpeza" });
	s.addText("10. Qualidade e reprodutibilidade", { placeholder: "title" });
	table(s, { y: 1.27, colW: [2.82, 5.07], header: ["Garantia", "Como"], rows: [
		["Código testado", "87 testes automáticos (pytest)"],
		["Mesmo arquivo em toda máquina", [t("SHA-256 do "), code("fer2013.csv"), t(" conferido pelo script")]],
		["Resultados comparáveis", "seed 42 e um único JSON de métricas"],
		["Sem dados no GitHub", "cada um baixa da fonte oficial"],
	] });
	block(s, { y: 3.55, h: 1.05, title: "Primeira linha de base: classe majoritária", body: [
		t("Prever sempre “alegria” acerta "), b("24,9%"), t(" da validação (F1 macro 0,06). É o piso para todos os métodos."),
	] });
	pageNumber(s, 10);
	s.addNotes("Primeiro número real do projeto: chutar sempre a classe mais comum dá 24,9% de acurácia. Qualquer método precisa ficar bem acima disso.");
}

pres.addSection({ title: "Encerramento" });

// 11. Próximos passos
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Encerramento" });
	s.addText("11. Próximos passos", { placeholder: "title" });
	s.addText([
		b("Frente B: ", { bullet: true }), t("pré-processamento (Viola-Jones, equalização, gaussiano) e os descritores Gabor, HOG, LBP e DCT", { breakLine: true }),
		b("Frente A: ", { bullet: true }), t("ligar o pipeline à base comum e treinar a CNN do zero e a ResNet-18", { breakLine: true }),
		b("CK+: ", { bullet: true }), t("aguardando a resposta dos autores; o código já está pronto e testado", { breakLine: true }),
		t("Revisão cruzada da base comum e merge na ", { bullet: true }), code("main"),
	], { x: M, y: 1.3, w: CW, h: 2.4, fontSize: 16, color: C.text1, paraSpaceAfter: 8, valign: "top", margin: 0,
		isTextBox: true, objectName: "Próximos passos" });
	block(s, { y: 3.95, h: 0.95, title: "Situação",
		body: [t("A base comum está pronta: as duas frentes já podem treinar sobre os mesmos dados.")] });
	pageNumber(s, 11);
	s.addNotes("Semana 2: a frente B parte para pré-processamento e descritores; a frente A conecta o pipeline dela à base comum e começa os treinos.");
}

// Encerramento
{
	const s = pres.addSlide({ masterName: "ENCERRAMENTO", sectionTitle: "Encerramento" });
	s.addText("Obrigado", { x: 0, y: 1.45, w: W, h: 0.8, fontSize: 40, bold: true, align: "center", color: C.background1,
		margin: 0, isTextBox: true, objectName: "Obrigado" });
	s.addText("Dúvidas e sugestões", { x: 0, y: 2.3, w: W, h: 0.4, fontSize: 17, bold: true, align: "center",
		color: C.background1, margin: 0, isTextBox: true, objectName: "Dúvidas" });
	const lh = 1.87, lw = lh * 3307 / 2480;
	s.addImage({ path: asset("INF-full-white.png"), x: W / 2 - lw / 2, y: 3.15, w: lw, h: lh,
		altText: "Logos do Instituto de Informática e da UFG", objectName: "Logos" });
}

(async () => {
	await pres.writeFile({ fileName: OUT });
	const { applyTheme } = require(process.env.PPTX_SKILL_THEME || "./apply_theme.js");
	await applyTheme(OUT, THEME);
	console.log("gerado:", OUT);
})();
