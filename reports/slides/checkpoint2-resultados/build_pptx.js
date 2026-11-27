// Checkpoint 2 (27/11): resultados preliminares e métricas de avaliação. Mesma identidade visual dos
// decks anteriores (tema UFG). Todos os números vêm de build/deck_data.json, gerado por assets.py a
// partir de results/*.json e splits/fer2013_clean.csv.
//
// Uso (da raiz do repositório):
//   python reports/slides/checkpoint2-resultados/assets.py
//   node reports/slides/checkpoint2-resultados/build_pptx.js      # precisa do pptxgenjs
"use strict";
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const HERE = __dirname;
const BUILD = path.join(HERE, "build");
const OUT = path.join(HERE, "checkpoint2-resultados.pptx");
const D = JSON.parse(fs.readFileSync(path.join(BUILD, "deck_data.json"), "utf8"));
const asset = (f) => path.join(BUILD, f);
const R = D.results;
const EMO = D.emotions;

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
const HEX = { blue: "0072B9", orange: "C76E00", green: "5B8D08", gray: "212121", muted: "6B6B6B", grid: "E3E3E3" };

// Formatação pt-BR
const pct = (v, d = 1) => (100 * v).toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }) + "%";
const dec = (v, d = 2) => v.toLocaleString("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
const int = (v) => v.toLocaleString("pt-BR");
const minutes = (s) => dec(s / 60, 1) + " min";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Checkpoint 2: resultados preliminares";
pres.subject = "INF0413 – Reconhecimento de emoções em imagens faciais";
pres.author = "Antonio H. Q. V. Lopes, Kauã L. Sandes, Luccas J. R. Clemente, Yago L. Coqueiro";
const C = pres.SchemeColor;

const W = 10, H = 5.625, M = 0.625, RIGHT = 8.51, CW = RIGHT - M;
const TOTAL = 7;
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

// ------------------------------------------------------------------ componentes do tema
const t = (s, o = {}) => ({ text: s, options: { ...o } });
const b = (s, o = {}) => ({ text: s, options: { bold: true, ...o } });
const code = (s, o = {}) => ({ text: s, options: { fontFace: "Consolas", ...o } });

function pageNumber(slide, n, white = false) {
	slide.addText(`${n}/${TOTAL}`, { x: W - 1.3, y: H - 0.36, w: 1.12, h: 0.28, fontSize: 12, bold: true, align: "right",
		color: white ? C.background1 : C.text1, margin: 0, isTextBox: true, objectName: "Número" });
}

function block(slide, { x = M, y, w = CW, h, title, body, kind = "block", size = 14 }) {
	const bar = { block: C.accent1, alert: C.accent4, example: C.accent3 }[kind];
	slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: C.background2 }, line: { type: "none" }, objectName: "Bloco fundo" });
	slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.088, h, fill: { color: bar }, line: { type: "none" }, objectName: "Bloco barra" });
	const runs = [];
	if (title) runs.push(b(title, { color: bar, breakLine: true }));
	for (const r of body) runs.push(r);
	slide.addText(runs, { x: x + 0.29, y, w: w - 0.45, h, fontSize: size, color: C.text1, valign: "middle", margin: 0,
		paraSpaceAfter: 2, isTextBox: true, objectName: title || "Bloco" });
}

function table(slide, { x = M, y, colW, header, rows, fontSize = 13.5, rowH }) {
	const none = { type: "none" };
	const thick = { type: "solid", pt: 1.25, color: HEX.gray };
	const thin = { type: "solid", pt: 0.75, color: HEX.gray };
	const cell = (content, opts) => ({ text: content, options: { margin: [3, 6, 3, 6], valign: "middle", fontSize, color: C.text1, ...opts } });
	const out = [header.map((h) => cell(h, { bold: true, border: [thick, none, thin, none] }))];
	rows.forEach((r, i) => {
		const last = i === rows.length - 1;
		out.push(r.map((c) => cell(c, { fill: { color: i % 2 ? C.background2 : C.background1 }, border: [none, none, last ? thick : none, none] })));
	});
	slide.addTable(out, { x, y, w: colW.reduce((a, v) => a + v, 0), colW, rowH, autoPage: false, objectName: "Tabela" });
}

const axisFont = { catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 12,
	valAxisLabelFontSize: 11, catAxisLabelColor: HEX.gray, valAxisLabelColor: HEX.muted };

// ------------------------------------------------------------------ capa
pres.addSection({ title: "Abertura" });
{
	const s = pres.addSlide({ masterName: "CAPA", sectionTitle: "Abertura" });
	s.addText("Reconhecimento de Emoções em Imagens Faciais", { x: 0.7, y: 0.78, w: 8.6, h: 0.7, fontSize: 30, bold: true,
		align: "center", color: C.text1, margin: 0, isTextBox: true, objectName: "Título" });
	s.addText("Checkpoint 2: resultados preliminares e métricas de avaliação", { x: 0.7, y: 1.5, w: 8.6, h: 0.36,
		fontSize: 15, bold: true, align: "center", color: C.text1, margin: 0, isTextBox: true, objectName: "Subtítulo" });
	[["Antonio H. Q. V. Lopes", "(202503993)", "Kauã L. Sandes", "(202504016)"],
	 ["Luccas J. R. Clemente", "(202504019)", "Yago L. Coqueiro", "(202504028)"]].forEach((row, i) => {
		const y = 2.15 + i * 0.4;
		s.addText([b(row[0]), t(" " + row[1])], { x: 0.6, y, w: 4.25, h: 0.36, fontSize: 17, align: "right", color: C.text1,
			margin: 0, isTextBox: true, objectName: `Autor ${2 * i + 1}` });
		s.addText([b(row[2]), t(" " + row[3])], { x: 5.2, y, w: 4.25, h: 0.36, fontSize: 17, align: "left", color: C.text1,
			margin: 0, isTextBox: true, objectName: `Autor ${2 * i + 2}` });
	});
	s.addText([t("INF0413 – Processamento de Sinal e Imagem · Prof. Aldo Díaz", { breakLine: true }),
		t("Instituto de Informática · Universidade Federal de Goiás", { breakLine: true }),
		t("github.com/KauaSandes/seminario-reconhecimento-emocoes")],
		{ x: 0.9, y: 3.15, w: 8.2, h: 0.72, fontSize: 12.5, align: "center", color: C.text1, margin: 0, isTextBox: true, objectName: "Instituição" });
	s.addText("Checkpoint 2 · 27 de novembro de 2026", { x: 2.0, y: 4.0, w: 5.4, h: 0.32, fontSize: 13.5, bold: true,
		align: "center", color: C.text1, margin: 0, isTextBox: true, objectName: "Data" });
	s.addNotes("Abertura (10 s). Hoje: primeiros resultados das duas frentes e como estamos medindo.");
}

pres.addSection({ title: "Resultados" });

// ------------------------------------------------------------------ 1. Onde estamos
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Resultados" });
	s.addText("1. Onde estamos", { placeholder: "title" });
	const rows = [
		{ label: "Fase 1\nPreparar os dados", color: C.text1, boxes: [["Bases", "FER-2013 oficial", 1], ["Limpeza", "duplicatas e vazamento", 1], ["Divisão", "treino, validação, teste", 1]] },
		{ label: "Fase 2a\nRedes neurais", color: C.accent1, boxes: [["CNN do zero", "treinada", 1], ["ResNet-18", "treinada", 1], ["Vídeo (CK+)", "extra: aguarda acesso", 0]] },
		{ label: "Fase 2b\nClássicos", color: C.accent2, boxes: [["Pré-processamento", "pronto", 1], ["Descritores", "HOG feito; LBP, Gabor, DCT", 0], ["Classificadores", "SVM feito; k-NN", 0]] },
		{ label: "Fase 3\nAvaliar", color: C.accent3, boxes: [["Métricas", "em uso", 1], ["Robustez", "código pronto", 0], ["Interpretação", "Grad-CAM, FFT2", 0]] },
	];
	const bx0 = 2.47, bw = 1.85, gap = 0.22, bh = 0.56;
	let y = 1.2;
	rows.forEach((r) => {
		s.addText(r.label, { x: M - 0.15, y, w: bx0 - M - 0.05, h: bh, fontSize: 12, bold: true, align: "right", valign: "middle",
			color: r.color, margin: 0, isTextBox: true, objectName: r.label.split("\n")[0] });
		r.boxes.forEach(([title, sub, done], i) => {
			const x = bx0 + i * (bw + gap);
			s.addText([b(title, { breakLine: true }), t(sub, { fontSize: 10.5 })], {
				shape: pres.shapes.ROUNDED_RECTANGLE, rectRadius: 0.06, x, y, w: bw, h: bh,
				fill: done ? { color: r.color } : { color: r.color, transparency: 82 },
				color: done ? C.background1 : r.color, fontSize: 12, align: "center", valign: "middle", margin: 2,
				line: { type: "none" }, objectName: title,
			});
		});
		y += bh + 0.13;
	});
	s.addText([b("cor cheia", { color: C.text1 }), t(" = feito    ·    "), b("cor clara", { color: C.text1 }), t(" = em andamento ou próximo passo")],
		{ x: bx0, y: y + 0.02, w: 3 * bw + 2 * gap, h: 0.3, fontSize: 11, color: HEX.muted, margin: 0, align: "left", isTextBox: true, objectName: "Legenda" });
	block(s, { y: 4.3, h: 0.62, body: [b(`${D.order.length} métodos já avaliados`), t(" com os mesmos dados e as mesmas métricas, só na validação.")] });
	pageNumber(s, 1);
	s.addNotes("(45 s) A Fase 1 está concluída e é comum às duas frentes. A Frente A treinou a CNN do zero e a ResNet-18; a Frente B já tem pré-processamento, HOG e um SVM. As métricas estão em uso; a robustez tem o código pronto e roda nos próximos passos. O vídeo com o CK+ virou extra porque o acesso à base ainda não chegou.");
}

// ------------------------------------------------------------------ 2. Dados
{
	const s = pres.addSlide({ masterName: "DESTAQUE_AZUL", sectionTitle: "Resultados" });
	s.addText("2. Dados prontos para todos", { placeholder: "title" });
	const dd = D.data;
	const stats = [[int(dd.train_after), `imagens de treino após a limpeza\n(eram ${int(dd.train_before)})`],
		[int(dd.leak), "cópias de imagens da validação\ne do teste tiradas do treino"],
		["0", `alterações na validação (${int(dd.val)})\ne no teste (${int(dd.test)})`]];
	const colw = (W - 2 * M) / 3;
	stats.forEach(([num, lab], i) => {
		const x = M + i * colw;
		s.addText(num, { x, y: 1.75, w: colw, h: 1.0, fontSize: 56, bold: true, align: "center", valign: "middle",
			color: C.background1, margin: 0, isTextBox: true, objectName: `Número ${i + 1}` });
		s.addText(lab, { x: x + 0.1, y: 2.85, w: colw - 0.2, h: 0.62, fontSize: 14, align: "center", valign: "top",
			color: C.background1, margin: 0, isTextBox: true, objectName: `Rótulo ${i + 1}` });
		if (i) s.addShape(pres.shapes.LINE, { x, y: 1.85, w: 0, h: 1.3, line: { color: "969696", width: 1 }, objectName: "Separador" });
	});
	s.addText("FER-2013 da competição oficial, conferida por SHA-256. As duas frentes usam exatamente os mesmos dados, a mesma limpeza e a mesma divisão.",
		{ x: M, y: 4.05, w: W - 2 * M, h: 0.62, fontSize: 14, color: C.background1, margin: 0, isTextBox: true, objectName: "Nota" });
	pageNumber(s, 2, true);
	s.addNotes(`(40 s) A limpeza tirou ${int(dd.removed)} imagens do treino: duplicatas, imagens vazias, rótulos conflitantes e, o mais importante, ${int(dd.leak)} cópias de imagens da validação e do teste, que inflariam a nota. Validação e teste ficaram intactos, para os números serem comparáveis com a literatura.`);
}

// ------------------------------------------------------------------ 3. Métricas
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Resultados" });
	s.addText("3. Como medimos", { placeholder: "title" });
	table(s, { y: 1.2, colW: [2.15, 5.74], header: ["Métrica", "O que responde"], rows: [
		[[b("Acurácia")], "Quantas imagens o modelo acerta. Engana quando as classes são desiguais."],
		[[b("F1 por classe")], "Para cada emoção: acerta quando prevê (precisão) e encontra as que existem (revocação)?"],
		[[b("F1 macro")], "Média dos 7 F1: cada emoção pesa igual, inclusive o nojo."],
		[[b("Matriz de confusão")], "Quais emoções o modelo troca entre si."],
	] });
	const maj = R.majority;
	block(s, { y: 3.68, h: 1.12, title: "Por que não só a acurácia", body: [
		t(`O nojo é só ${pct(D.data.disgust_share)} do treino. Prever sempre “alegria” dá `), b(pct(maj.accuracy)),
		t(" de acurácia, mas F1 macro de só "), b(dec(maj.f1_macro)), t(". Tudo medido na validação; o teste é usado uma única vez, no final."),
	] });
	pageNumber(s, 3);
	s.addNotes("(55 s) As quatro métricas saem do mesmo módulo para todos os métodos e são gravadas num JSON padronizado. A acurácia sozinha engana: a linha de base que só responde 'alegria' acerta um quarto, mas não reconhece nada, e o F1 macro mostra isso. Por isso a nossa métrica principal de comparação é o F1 macro, acompanhado da acurácia.");
}

// ------------------------------------------------------------------ 4. Resultados
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Resultados" });
	s.addText("4. Resultados preliminares (validação)", { placeholder: "title" });
	const labels = D.order.map((k) => R[k].label);
	s.addChart(pres.charts.BAR, [
		{ name: "Acurácia", labels, values: D.order.map((k) => R[k].accuracy) },
		{ name: "F1 macro", labels, values: D.order.map((k) => R[k].f1_macro) },
	], {
		x: M, y: 1.05, w: CW, h: 2.75, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
		chartColors: [HEX.blue, HEX.orange], showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00",
		dataLabelFontSize: 11, dataLabelColor: HEX.gray, dataLabelFontFace: "+mn-lt",
		valAxisMinVal: 0, valAxisMaxVal: 1, valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" },
		showLegend: true, legendPos: "t", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.gray,
		...axisFont, objectName: "Acurácia e F1 macro por método",
	});
	const res = R.resnet18, cnn = R.cnn_scratch, hog = R.hog_linearsvc;
	block(s, { y: 3.9, h: 0.72, size: 14, body: [
		t("As redes superam o HOG + SVM em mais de 20 pontos; a "), b("ResNet-18"),
		t(` (pré-treinada no ImageNet) é a melhor até agora: ${pct(res.accuracy)} e F1 macro ${dec(res.f1_macro)}.`),
	] });
	s.addText(`Redes (Frente A): RTX 5060, precisão mista, seed 42, ${minutes(cnn.train_time_s)} e ${minutes(res.train_time_s)} de treino. ` +
		`HOG + SVM (Frente B): CPU, ${minutes(hog.train_time_s)}. Referência: Khaireddin e Chen (2021), 73,3% no teste.`,
		{ x: M, y: 4.7, w: CW, h: 0.4, fontSize: 10.5, color: HEX.muted, margin: 0, isTextBox: true, objectName: "Rodapé" });
	pageNumber(s, 4);
	const pix = R.pixels_linearsvc;
	const pixNote = pix ? ` Com o mesmo SVM sobre os pixels brutos, fica em ${pct(pix.accuracy)}: o descritor HOG ganha ${dec(100 * (hog.accuracy - pix.accuracy), 1)} pontos.` : "";
	s.addNotes(`(65 s) Todos os números são da validação. Linha de base: ${pct(R.majority.accuracy)}. O primeiro clássico, HOG + SVM linear, chega a ${pct(hog.accuracy)} (F1 ${dec(hog.f1_macro)}).${pixNote} A CNN do zero chega a ${pct(cnn.accuracy)} e a ResNet-18 a ${pct(res.accuracy)} (F1 ${dec(res.f1_macro)}). A transferência do ImageNet ajuda mesmo em imagens cinza de 48×48. A referência da literatura é 73,3% no teste, com uma VGG ajustada com cuidado.`);
}

// ------------------------------------------------------------------ 5. F1 por classe
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Resultados" });
	s.addText("5. Desempenho por emoção (F1)", { placeholder: "title" });
	const series = ["hog_linearsvc", "cnn_scratch", "resnet18"].filter((k) => R[k]);
	const colors = { hog_linearsvc: HEX.orange, cnn_scratch: HEX.green, resnet18: HEX.blue };
	s.addChart(pres.charts.BAR, series.map((k) => ({ name: R[k].label, labels: EMO, values: R[k].f1_per_class })), {
		x: M, y: 1.05, w: CW, h: 2.85, barDir: "col", barGrouping: "clustered", barGapWidthPct: 45,
		chartColors: series.map((k) => colors[k]), valAxisMinVal: 0, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0",
		valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" },
		showLegend: true, legendPos: "t", legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.gray,
		showValAxisTitle: true, valAxisTitle: "F1", valAxisTitleFontSize: 11, valAxisTitleColor: HEX.muted,
		...axisFont, objectName: "F1 por emoção",
	});
	const f = (k, c) => dec(R[k].f1_per_class[c]);
	block(s, { y: 4.02, h: 0.98, title: "Nojo é o ponto fraco", body: [
		t(`A CNN do zero nunca acerta o nojo (F1 ${f("cnn_scratch", 1)}); a ResNet-18 chega a ${f("resnet18", 1)}. `),
		t(`Alegria (${f("resnet18", 3)}) e surpresa (${f("resnet18", 5)}) são as mais fáceis.`),
	] });
	pageNumber(s, 5);
	s.addNotes("(55 s) O F1 por classe explica o F1 macro. Alegria e surpresa têm mudanças grandes na boca e nos olhos, e todos os métodos vão bem. O nojo, com pouco mais de 1% do treino, é o problema: a CNN do zero nunca o prevê. O próximo passo é treinar com os pesos de classe que já existem na base comum.");
}

// ------------------------------------------------------------------ 6. Matriz de confusão
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Resultados" });
	s.addText("6. Onde a ResNet-18 erra", { placeholder: "title" });
	const cm = R.resnet18.confusion_matrix;
	const abbr = ["raiva", "nojo", "medo", "alegria", "tristeza", "surpresa", "neutro"];
	const ramp = (f) => {   // branco -> UFGBlue (sequencial, um tom)
		const a = [255, 255, 255], z = [0, 114, 185];
		return a.map((v, i) => Math.round(v + (z[i] - v) * f).toString(16).padStart(2, "0")).join("").toUpperCase();
	};
	const head = [{ text: "", options: { fontSize: 9 } },
		...abbr.map((a) => ({ text: a, options: { fontSize: 10, bold: true, align: "center", color: HEX.gray } }))];
	const rows = [head];
	cm.forEach((row, i) => {
		const tot = row.reduce((a, v) => a + v, 0);
		rows.push([{ text: abbr[i], options: { fontSize: 10.5, bold: true, color: HEX.gray } },
			...row.map((v) => {
				const fr = v / tot, shade = Math.min(fr / 0.75, 1);
				return { text: int(v), options: { fontSize: 10.5, align: "center",
					fill: { color: ramp(shade) }, color: shade > 0.5 ? "FFFFFF" : HEX.gray } };
			})]);
	});
	const none = { type: "none" };
	rows.forEach((r) => r.forEach((c) => { c.options.border = [none, none, none, none]; c.options.margin = [1, 3, 1, 3]; c.options.valign = "middle"; }));
	s.addTable(rows, { x: M, y: 1.18, w: 5.05, colW: [0.98, ...Array(7).fill(0.581)], rowH: 0.36, autoPage: false, objectName: "Matriz de confusão" });
	// três maiores confusões, fora da diagonal
	const off = [];
	cm.forEach((row, i) => row.forEach((v, j) => { if (i !== j) off.push([v, i, j]); }));
	off.sort((a, b2) => b2[0] - a[0]);
	const top = off.slice(0, 3).map(([v, i, j]) => `${EMO[i]} → ${EMO[j]}: ${int(v)}`);
	const nojoTot = cm[1].reduce((a, v) => a + v, 0);
	block(s, { x: 5.9, y: 1.18, w: RIGHT - 5.9, h: 2.88, size: 13.5, title: "Maiores confusões", body: [
		t(top[0], { bullet: true, breakLine: true }), t(top[1], { bullet: true, breakLine: true }), t(top[2], { bullet: true, breakLine: true }),
		t(`nojo → raiva: ${int(cm[1][0])} de ${int(nojoTot)}`, { bullet: true }),
	] });
	s.addText("Linhas: emoção real; colunas: emoção prevista; cor mais escura = maior fração da linha. Expressões sutis e parecidas em 48×48 pixels são as mais trocadas, como na literatura.",
		{ x: M, y: 4.25, w: CW, h: 0.55, fontSize: 12, color: HEX.muted, margin: 0, isTextBox: true, objectName: "Leitura" });
	pageNumber(s, 6);
	s.addNotes("(50 s) Linhas são a emoção real e colunas a prevista; a diagonal são os acertos. As confusões principais são entre tristeza, neutro e medo, que diferem pouco numa imagem de 48 por 48. O nojo é confundido sobretudo com raiva. É o padrão relatado na literatura da FER-2013.");
}

// ------------------------------------------------------------------ 7. Próximos passos
pres.addSection({ title: "Encerramento" });
{
	const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: "Encerramento" });
	s.addText("7. Próximos passos", { placeholder: "title" });
	table(s, { y: 1.2, colW: [1.7, 6.19], header: ["Quem", "Próximo passo"], rows: [
		[[b("Frente A")], "Treinar com pesos de classe (nojo); Grad-CAM e FFT2 dos filtros aprendidos"],
		[[b("Frente B")], "Descritores LBP, Gabor e DCT; k-NN; ajuste do SVM na validação"],
		[[b("As duas")], "Robustez (ruído, desfoque, resolução) e avaliação final no teste, uma vez por método"],
	] });
	block(s, { y: 3.35, h: 0.95, title: "Situação", body: [
		t(`${D.order.length} métodos avaliados com os mesmos dados e as mesmas métricas. O teste segue intocado para a avaliação final.`),
	] });
	pageNumber(s, 7);
	s.addNotes("(40 s) Próximas semanas: balancear as classes, completar os descritores clássicos, rodar a robustez e, por fim, a avaliação única no teste.\n\n" +
		"PERGUNTAS PROVÁVEIS:\n" +
		"• Por que validação e não teste? O teste é usado uma única vez, no final, para não ajustarmos nada a ele.\n" +
		"• Por que o nojo vai mal? Só 1,3% do treino; vamos usar os pesos de classe (nojo pesa cerca de 11 vezes mais).\n" +
		"• Por que a ResNet ganha? Ela já aprendeu bordas e texturas no ImageNet; a CNN do zero aprende tudo só com 27 mil imagens.\n" +
		"• Por que o HOG + SVM fica abaixo? O SVM é linear e o HOG captura só gradientes locais; faltam LBP, Gabor e ajuste.\n" +
		"• De onde vêm os dados? Competição oficial do Kaggle; o SHA-256 confere que todas as máquinas têm o mesmo arquivo.");
}

// ------------------------------------------------------------------ encerramento
{
	const s = pres.addSlide({ masterName: "ENCERRAMENTO", sectionTitle: "Encerramento" });
	s.addText("Obrigado", { x: 0, y: 1.45, w: W, h: 0.8, fontSize: 40, bold: true, align: "center", color: C.background1, margin: 0, isTextBox: true, objectName: "Obrigado" });
	s.addText("Dúvidas e sugestões", { x: 0, y: 2.3, w: W, h: 0.4, fontSize: 17, bold: true, align: "center", color: C.background1, margin: 0, isTextBox: true, objectName: "Dúvidas" });
	const lh = 1.87, lw = lh * 3307 / 2480;
	s.addImage({ path: asset("INF-full-white.png"), x: W / 2 - lw / 2, y: 3.15, w: lw, h: lh, altText: "Logos do Instituto de Informática e da UFG", objectName: "Logos" });
	s.addNotes("Abrir para as perguntas (até 2).");
}

(async () => {
	await pres.writeFile({ fileName: OUT });
	const { applyTheme } = require(process.env.PPTX_SKILL_THEME || "./apply_theme.js");
	await applyTheme(OUT, THEME);
	console.log("gerado:", OUT);
})();
