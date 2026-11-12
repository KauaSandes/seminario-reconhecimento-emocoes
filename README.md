# INF0413 - Seminário

## Título do Seminário
**Reconhecimento de Emoções em Imagens Faciais: um pipeline de processamento de imagem com filtros, convolução e Fourier**

## Equipe:
- 202503993 - ANTONIO HENRIQUE QUEIROZ VILLAR LOPES
- 202504016 - KAUÃ LOPES SANDES
- 202504019 - LUCCAS JOSE RODRIGUES CLEMENTE
- 202504028 - YAGO LIMA COQUEIRO

## Apresentações
1. Apresentação 1 – Tema e justificativa: [[link dos slides]](https://canva.link/ul9vfvoujlhpmmj)
2. Apresentação 2 – (a definir)

## Resumo
Expressões faciais são um dos principais canais de comunicação de emoções entre pessoas, e reconhecê-las automaticamente tem aplicações em saúde mental, acessibilidade e interação humano-computador.
Uma imagem de rosto é um sinal bidimensional, no qual as emoções se manifestam como padrões de bordas, texturas e frequências espaciais.
Este projeto investiga como as ferramentas de processamento de sinais e imagens podem extrair esses padrões.
Construiremos um pipeline com pré-processamento (escala de cinza, equalização e filtragem), detecção do rosto e extração de características por filtros de Gabor, gradientes (HOG), LBP e coeficientes da DCT/FFT2.
As características serão classificadas em sete emoções básicas com classificadores simples (SVM e k-NN), usando as bases públicas FER-2013 e CK+.
Também avaliaremos o efeito de ruído, desfoque e redução de resolução sobre o reconhecimento.
Esperamos comparar quantitativamente os extratores de características por acurácia, F1 e matriz de confusão.
Esperamos ainda identificar quais faixas de frequência espacial concentram a informação emocional do rosto.
Ao final, o projeto deve mostrar que conceitos como convolução 2D e transformada de Fourier são a base de uma aplicação real de visão computacional.

## Introdução
- **Motivação:** As emoções orientam a comunicação humana, e o rosto é sua principal vitrine. Sistemas que reconhecem expressões faciais já são usados em saúde, educação, segurança e interfaces adaptativas. Por trás desses sistemas estão operações clássicas de processamento de imagens: filtragem, detecção de bordas e análise em frequência, as mesmas estudadas na disciplina.
- **Revisão bibliográfica:** (em construção, mínimo de 5 fontes) Ekman e Friesen (emoções básicas); Viola e Jones (detecção de rostos); Lyons et al. (Gabor e base JAFFE); Shan, Gong e McOwan (LBP para expressões); Dalal e Triggs (HOG); Goodfellow et al. (FER-2013); Lucey et al. (CK+).
- **Objetivos:**
  - Entender e aplicar convolução 2D, filtros de Gabor, gradientes e transformadas (FFT2/DCT) na extração de características faciais.
  - Comparar extratores de características para o reconhecimento de 7 emoções.
  - Analisar a robustez do reconhecimento a ruído, desfoque e baixa resolução (amostragem).

## Metodologia
- **Conjunto de dados (*Dataset*):** FER-2013 (~35 mil imagens 48×48 em escala de cinza, 7 classes) e CK+ (sequências posadas em laboratório), extraídas da literatura. Opcionalmente, fotos da própria equipe para demonstração. A AED incluirá distribuição de classes, imagens médias por emoção e espectros médios (FFT2).
- **Modelos:** Python, NumPy, SciPy, OpenCV, scikit-image e scikit-learn em Jupyter/Colab.
  1. Pré-processamento: escala de cinza, equalização de histograma, filtro gaussiano.
  2. Detecção e recorte do rosto: Viola-Jones (Haar cascades).
  3. Características: banco de filtros de Gabor, HOG, LBP, coeficientes DCT/FFT2.
  4. Classificação: SVM e k-NN.
- **Métricas:** acurácia, F1 por classe, matriz de confusão, tempo de processamento (convolução espacial vs. via FFT).
- **Fluxograma:**

```
Imagem -> Pré-processamento -> Detecção do rosto -> Extração de características -> Classificação -> Emoção
          (cinza, equalização,   (Viola-Jones)       (Gabor, HOG, LBP, DCT/FFT2)    (SVM, k-NN)
           filtro gaussiano)
```

### Cronograma
| Semana | Atividade |
|---|---|
| 1 | Revisão bibliográfica e análise exploratória dos dados |
| 2 | Pré-processamento e detecção do rosto |
| 3 | Extração de características |
| 4 | Classificação e experimentos de robustez |
| 5 | Análise dos resultados e apresentação final |

## Resultados
(em desenvolvimento)

## Discussão
(em desenvolvimento)

## Conclusão
(em desenvolvimento)

## Referências
(em construção, formato ABNT)
