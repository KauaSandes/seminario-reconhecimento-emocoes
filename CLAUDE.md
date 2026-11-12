# CLAUDE.md — Seminário INF0413: Reconhecimento de emoções em imagens faciais

Contexto para o Claude Code. Leia antes de qualquer tarefa neste repositório.

## 1. O projeto

Trabalho final da disciplina **INF0413 – Processamento de Sinal e Imagem** (UFG, 2026-2, Prof. Aldo Díaz).

- **Tema:** reconhecer 7 emoções em rostos (raiva, nojo, medo, alegria, tristeza, surpresa, neutro).
- **Foco (pedido do professor):** redes neurais em **imagem e vídeo**. Os **métodos clássicos baseados em descritores** entram como **comparação**.
- **Ligação com a disciplina:** convolução 2D (filtros aprendidos × filtros projetados à mão), análise em frequência (FFT2/DCT) e testes de ruído e amostragem, que vêm dos labs le03 (ruído) e le04 (amostragem e FFT).
- **Repositório:** github.com/KauaSandes/seminario-reconhecimento-emocoes
- **Equipe:** Antonio H. Q. V. Lopes (202503993), Kauã L. Sandes (202504016), Luccas J. R. Clemente (202504019), Yago L. Coqueiro (202504028).
- **Apresentação aprovada no checkpoint (10 slides):** https://www.canva.com/design/DAHaEkIrbHY/gXNcynENYDrUaqiH7J4_mA/edit

## 2. Divisão da equipe

O trabalho corre em **duas frentes paralelas** sobre os **mesmos dados e a mesma avaliação**.

| Frente | Pessoas | Responsável por |
|---|---|---|
| **A: Redes neurais** | Antonio e Kauã | `src/cnn/`: CNN do zero, transfer learning (ResNet-18), vídeo (CNN + LSTM), Grad-CAM, FFT2 dos filtros aprendidos, demo na webcam |
| **B: Métodos clássicos** | Yago e Luccas | `src/classic/`: pré-processamento, descritores (Gabor, HOG, LBP, DCT), SVM e k-NN. Também cuidam da base comum `src/common/`: limpeza, divisão, degradações e métricas |

A frente B cuida de `src/common/` porque os métodos clássicos usam menos GPU, e a frente A já tem mais itens. **Mudanças em `src/common/` precisam de revisão de alguém das duas frentes**, porque elas afetam a comparação.

## 3. Metodologia (resumo dos slides)

```
Fase 1 (comum)    Bases → Limpeza (duplicatas, rótulos, sem rosto) → Divisão treino/val/teste
Fase 2a (A)       CNN do zero 48×48 → Transfer learning ResNet-18 → Vídeo: CNN por quadro + LSTM
Fase 2b (B)       Pré-processamento (Viola-Jones, equalização, gaussiano) → Gabor/HOG/LBP/DCT → SVM, k-NN
Fase 3 (comum)    Métricas (acurácia, F1, matriz) → Robustez (ruído, desfoque, resolução) → Interpretação (Grad-CAM, FFT2)
```

### Frente A: redes neurais (Antonio e Kauã)
- **CNN do zero (FER-2013):** 3 blocos conv 3×3 + BatchNorm + ReLU + MaxPool, depois dropout e camada densa. Entrada 48×48 em cinza. Aumento de dados: espelhamento horizontal, rotação de ±10°, recorte aleatório. Otimizador Adam e parada antecipada pela perda de validação.
- **Transfer learning (FER-2013):** ResNet-18 do torchvision com pesos do ImageNet. Redimensionar para 224×224 (112×112 se faltar tempo) e repetir o canal cinza em 3 canais com a normalização do ImageNet. Primeiro treinar só a camada final, depois a rede inteira com taxa de aprendizado menor. Usar precisão mista (`torch.autocast`).
- **Vídeo (CK+):** sequências do rosto neutro até o pico. Uma CNN pré-treinada e congelada (a do item acima) gera um vetor por quadro, e uma LSTM classifica a sequência. Como o CK+ é pequeno, extrair os vetores uma vez e salvar em disco.
- **Interpretação:** Grad-CAM (pacote `pytorch-grad-cam`) e FFT2 dos filtros da primeira camada, para comparar com os filtros de Gabor da frente B.
- **Demo:** webcam com OpenCV (detecção do rosto com Haar cascade e depois o melhor modelo).

### Frente B: métodos clássicos (Yago e Luccas)
- **Pré-processamento:** Viola-Jones (`cv2.CascadeClassifier`) **só no CK+ e nas fotos**, porque a FER-2013 já vem recortada. Depois escala de cinza, equalização de histograma e filtro gaussiano leve.
- **Descritores:** banco de filtros de Gabor (`skimage.filters.gabor_kernel`, várias orientações e escalas, com média e variância por resposta), HOG (`skimage.feature.hog`), LBP uniforme com histogramas por bloco (`skimage.feature.local_binary_pattern`) e DCT 2D (`scipy.fft.dctn`, coeficientes de baixa frequência em zigue-zague).
- **Classificadores:** SVM (começar com `LinearSVC`; RBF só depois de reduzir a dimensão com PCA, porque são 28 mil amostras) e k-NN. Padronizar as características com `StandardScaler` dentro de um `Pipeline`. Ajustar os hiperparâmetros **só na validação**.
- **Base comum (`src/common/`):** limpeza da FER-2013 (duplicatas por hash, imagens sem rosto ou constantes, desbalanceamento), arquivos de divisão, funções de degradação e métricas.

## 4. Dados

- **FER-2013:** `fer2013.csv` com as colunas `emotion`, `pixels` (2304 valores separados por espaço) e `Usage`. O valor `Training` vai para treino, `PublicTest` para validação e `PrivateTest` para teste. São 28.709, 3.589 e 3.589 imagens, num total de 35.887.
  - Rótulos: 0 raiva, 1 nojo, 2 medo, 3 alegria, 4 tristeza, 5 surpresa, 6 neutro. **Essa ordem é a oficial do projeto.**
- **CK+:** o acesso é pedido aos autores (Lucey et al., 2010). São 593 sequências de 123 pessoas, das quais 327 têm rótulo. Rótulos originais: 1 raiva, 2 desprezo, 3 nojo, 4 medo, 5 alegria, 6 tristeza, 7 surpresa.
  - **Descartar "desprezo"** (fora das 7 classes) e mapear o resto para a ordem da FER-2013. Para a classe neutro, usar o primeiro quadro das sequências.
  - Validação cruzada em **10 partes agrupadas por pessoa** (`GroupKFold` com o ID da pessoa). A mesma pessoa nunca aparece no treino e no teste ao mesmo tempo.
- **Nunca versionar dados nem pesos.** `data/`, `checkpoints/` e `*.pt` ficam no `.gitignore`. Os pesos vão para o Google Drive.
- Os **arquivos de divisão** (`splits/*.csv`) **são versionados**, para todos usarem exatamente as mesmas amostras.

## 5. Contrato entre as frentes (o que garante uma comparação justa)

1. Os dados vêm **só** de `src/common/data.py` (`load_fer2013(split)` e `load_ckplus(fold)`), já limpos e divididos.
2. As degradações vêm **só** de `src/common/degrade.py`: `add_gaussian_noise(img, sigma)` com σ ∈ {5, 10, 20}, `gaussian_blur(img, ksize)` com ksize ∈ {3, 5, 7} e `downsample(img, size)` para 24 e 12 px, voltando a 48 px. As degradações são aplicadas **só no teste**.
3. As métricas vêm **só** de `src/common/metrics.py`, e cada execução grava um JSON em `results/`:
   ```json
   {"method": "resnet18", "track": "cnn", "dataset": "fer2013", "split": "test",
    "degradation": "noise", "level": 10, "accuracy": 0.0, "f1_macro": 0.0,
    "f1_per_class": [0,0,0,0,0,0,0], "confusion_matrix": [[...]],
    "train_time_s": 0.0, "inference_ms_per_img": 0.0, "seed": 42, "git_commit": "..."}
   ```
   Para imagem sem degradação, usar `degradation: "none"` e `level: 0`.
4. **O conjunto de teste é usado uma única vez por método**, no final. Durante o desenvolvimento, usar a validação.
5. Usar `seed = 42` em tudo (`random`, `numpy`, `torch`, `cudnn.deterministic` quando possível).
6. Linhas de base para comparar: classe majoritária e pixels brutos + SVM (frente B).

## 6. Estrutura do repositório

```
seminario-reconhecimento-emocoes/
├── CLAUDE.md
├── README.md                # relatório do seminário (resumo, metodologia, resultados, referências)
├── requirements.txt
├── .gitignore               # data/, checkpoints/, *.pt, __pycache__/
├── splits/                  # versionado: fer2013_clean.csv, ckplus_folds.csv
├── src/
│   ├── common/              # frente B mantém, as duas frentes revisam
│   │   ├── data.py  clean.py  degrade.py  metrics.py  utils.py
│   ├── cnn/                 # frente A
│   │   ├── models.py  train_scratch.py  train_resnet.py
│   │   ├── video_lstm.py  gradcam.py  filters_fft.py  webcam_demo.py
│   └── classic/             # frente B
│       ├── preprocess.py  features.py  train_classic.py
├── notebooks/               # AED e figuras, com prefixo 01_, 02_, ...
├── results/                 # JSONs de métricas (versionados)
└── reports/figures/         # figuras do relatório e dos slides
```

## 7. Cronograma (adaptado às duas frentes)

| Semana | Período | Frente A (Antonio e Kauã) | Frente B (Yago e Luccas) |
|---|---|---|---|
| 1 | 06–12/11 | Ambiente PyTorch na RTX 5060 e revisão de redes | Limpeza da FER-2013, `splits/`, `src/common/` |
| 2 | 13–19/11 | CNN do zero e ResNet-18 na FER-2013 | Pré-processamento e os 4 descritores |
| 3 | 20–26/11 | Vídeo no CK+ (CNN + LSTM) e demo na webcam | SVM e k-NN, ajuste na validação, linhas de base |
| 4 | 27/11–03/12 | Grad-CAM, FFT2 dos filtros, testes de robustez | Testes de robustez e tabela comparativa |
| 5 | 04–10/12 | Teste final, análise conjunta, slides e relatório | Teste final, análise conjunta, slides e relatório |

O **checkpoint 1** é em 13/11. A data da apresentação final ainda precisa ser confirmada.

## 8. Infraestrutura

- **GPU principal:** NVIDIA RTX 5060 com 8 GB, de um dos membros. Se ela estiver com alguém da frente B, essa pessoa roda os treinos da frente A ou empresta a máquina.
  - A série RTX 50 (Blackwell) exige **PyTorch compilado com CUDA 12.8 ou mais novo**. Confirmar com `torch.cuda.is_available()`. O erro "no kernel image is available" indica versão antiga.
- **Reserva:** Google Colab ou Kaggle (GPU T4). Salvar checkpoints no Drive a cada época.
- **Frente B:** só CPU e pelo menos 16 GB de RAM para o SVM com HOG.
- **Bibliotecas:** python 3.11+, torch e torchvision (cu128+), opencv-python, scikit-image, scikit-learn, scipy, numpy, pandas, matplotlib, pytorch-grad-cam, tqdm.

## 9. Convenções para o Claude Code

- Nomes de variáveis e funções em inglês. Comentários, docstrings, README e mensagens de commit podem ser em português.
- Cada script de treino recebe argumentos por `argparse`, aceita `--seed` e `--device` e termina gravando o JSON de métricas pelo `src/common/metrics.py`.
- Não duplicar carregamento de dados, degradações ou métricas dentro de `src/cnn/` ou `src/classic/`. Importar de `src/common/`.
- Fluxo no Git: um branch por tarefa (`cnn/resnet18`, `classic/hog-svm`, `common/splits`), Pull Request para a `main`, e revisão de alguém da outra frente quando o PR mexer em `src/common/`.
- Ao gerar figuras para os slides, salvar em `reports/figures/` com nome descritivo.
- Se pedirem uma tarefa de uma frente, consultar o contrato da seção 5 antes de alterar qualquer interface compartilhada.
- Não se dê coautoria em nenhum commit no repositório do github.

## 10. Referências principais

- **Redes:** Li e Deng (2022), revisão de FER profunda; Simonyan e Zisserman (2015), VGG; He et al. (2016), ResNet; Mollahosseini et al. (2016); Khaireddin e Chen (2021), VGG com 73,28% na FER-2013; Donahue et al. (2015), LRCN; Tran et al. (2015), C3D; Jiang et al. (2020), DFEW; Selvaraju et al. (2017), Grad-CAM.
- **Clássicos e bases:** Ahmed et al. (1974), DCT; Lyons et al. (1998), Gabor; Viola e Jones (2004); Dalal e Triggs (2005), HOG; Shan et al. (2009), LBP; Ekman e Friesen (1971); Lucey et al. (2010), CK+; Goodfellow et al. (2015), FER-2013; Gonzalez e Woods (2010).