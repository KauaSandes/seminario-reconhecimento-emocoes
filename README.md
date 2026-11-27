# INF0413 - Seminário

## Título do Seminário
**Reconhecimento de Emoções em Imagens Faciais: um pipeline de processamento de imagem com filtros, convolução e Fourier**

## Equipe
- 202503993 - ANTONIO HENRIQUE QUEIROZ VILLAR LOPES
- 202504016 - KAUÃ LOPES SANDES
- 202504019 - LUCCAS JOSE RODRIGUES CLEMENTE
- 202504028 - YAGO LIMA COQUEIRO

| Frente | Membros | O que faz |
|---|---|---|
| **A: redes neurais** | Antonio (demo ao vivo) e Kauã (treinamento) | CNN do zero, ResNet-18 com transferência de aprendizado, Grad-CAM, FFT2 dos filtros aprendidos, demo na webcam |
| **B: métodos clássicos** | Yago e Luccas | Base comum de dados (`src/common/`), descritores Gabor, HOG, LBP e DCT, classificadores SVM e k-NN |

## Apresentações
1. Apresentação 1 – Tema e justificativa: [[link dos slides]](https://canva.link/ul9vfvoujlhpmmj)
2. Checkpoint 1 (13/11) – Motivação, contexto, metodologia e cronograma: slides no Canva (link acima); slides do ambiente da Frente A em [`reports/slides/checkpoint1-ambiente/`](reports/slides/checkpoint1-ambiente/)
3. Checkpoint 2 – Preparação e limpeza dos dados: [`reports/slides/checkpoint2-dados/`](reports/slides/checkpoint2-dados/)
4. Apresentação final – Resultados e **demonstração ao vivo** na webcam: (em preparação)

## Andamento

*Atualizado em 27/11/2026, início da semana 4.*

| Etapa | Situação |
|---|---|
| Base comum: download, limpeza, divisão, degradações e métricas (`src/common/`) | ✅ Pronta |
| CNN do zero e ResNet-18 na FER-2013 | ✅ Treinadas (validação) |
| Demo na webcam | 🟡 Implementada; falta testar com câmera de verdade e preparar para a apresentação |
| HOG + SVM linear e linhas de base | ✅ Prontos (branch `classic/hog-svm`) |
| Gabor, LBP, DCT, k-NN e SVM RBF | ⏳ Semana 4 |
| Balanceamento de classes, Grad-CAM e FFT2 dos filtros | ⏳ Semana 4 |
| Testes de robustez (ruído, desfoque, resolução) | ⏳ Semana 4, na validação |
| Avaliação final no teste | ⏳ Semana 5, uma única vez por método |
| Vídeo no CK+ (CNN + LSTM) | ⏸️ Extra; o acesso ao CK+ ainda não chegou |

## Resumo
Expressões faciais são um dos principais canais de comunicação de emoções entre pessoas, e reconhecê-las automaticamente tem aplicações em saúde mental, acessibilidade e interação humano-computador.
Uma imagem de rosto é um sinal bidimensional, no qual as emoções se manifestam como padrões de bordas, texturas e frequências espaciais.
Este projeto investiga como as ferramentas de processamento de sinais e imagens extraem esses padrões, comparando duas abordagens sobre os mesmos dados.
A principal usa redes neurais convolucionais: uma CNN treinada do zero e uma ResNet-18 pré-treinada no ImageNet, nas quais os filtros de convolução 2D são aprendidos a partir dos dados.
A abordagem de comparação usa métodos clássicos: pré-processamento (equalização e filtro gaussiano), detecção do rosto por Viola-Jones e descritores projetados à mão (filtros de Gabor, HOG, LBP e DCT), classificados com SVM e k-NN.
As duas abordagens são avaliadas nas sete emoções básicas da FER-2013 (com o CK+ como base complementar), pelas mesmas métricas: acurácia, F1 por classe e macro, matriz de confusão e tempo.
Também avaliamos a robustez a ruído, desfoque e redução de resolução, e analisamos os filtros aprendidos pela rede no domínio da frequência (FFT2), comparando-os com os filtros de Gabor.
Ao final, o projeto deve mostrar que convolução 2D e transformada de Fourier são a base tanto dos métodos clássicos quanto das redes neurais modernas, com uma demonstração ao vivo na webcam.

## Introdução
- **Motivação:** As emoções orientam a comunicação humana, e o rosto é sua principal vitrine. Sistemas que reconhecem expressões faciais já são usados em saúde, educação, segurança e interfaces adaptativas. Por trás desses sistemas estão operações de processamento de imagens estudadas na disciplina: filtragem, detecção de bordas, convolução e análise em frequência.
- **Revisão bibliográfica:** Ekman e Friesen (emoções básicas); Viola e Jones (detecção de rostos); Lyons et al. (Gabor); Shan, Gong e McOwan (LBP); Dalal e Triggs (HOG); Ahmed, Natarajan e Rao (DCT); He et al. (ResNet); Selvaraju et al. (Grad-CAM); Goodfellow et al. (FER-2013); Lucey et al. (CK+); Khaireddin e Chen (estado da arte na FER-2013). Lista completa em [Referências](#referências).
- **Objetivos:**
  - Treinar redes neurais convolucionais para reconhecer 7 emoções e comparar com descritores clássicos sob as mesmas condições.
  - Entender e aplicar convolução 2D, filtros de Gabor, gradientes e transformadas (FFT2/DCT) na extração de características faciais.
  - Analisar a robustez do reconhecimento a ruído, desfoque e baixa resolução (amostragem).
  - Interpretar o que a rede aprendeu: mapas de Grad-CAM e espectro dos filtros da primeira camada.
  - Demonstrar o modelo ao vivo, na webcam, na apresentação final.

## Metodologia

```
                         ┌─ Frente A: CNN do zero (48×48) / ResNet-18 (224×224) ──────────┐
Imagem -> Pré-processamento┤                                                                ├-> Emoção -> Métricas
 (FER-2013,  e limpeza     └─ Frente B: Viola-Jones -> Gabor / HOG / LBP / DCT -> SVM / k-NN ┘    (mesmo contrato)
  CK+)      (src/common)
```

- **Dados:** FER-2013 (35.887 imagens 48×48 em cinza, 7 classes, divisão oficial) e CK+ (sequências posadas em laboratório). Detalhes na seção [Dados](#dados).
- **Frente A (redes neurais):** PyTorch na RTX 5060, com precisão mista.
  - `SmallCNN`: 3 blocos convolucionais (64, 128 e 256 canais), 3,5 M de parâmetros, entrada 48×48.
  - ResNet-18 com pesos do ImageNet: primeiro só a última camada, depois a rede inteira.
  - Adam com parada antecipada pela perda de validação; seed 42.
  - Interpretação: Grad-CAM e FFT2 dos filtros da primeira camada.
- **Frente B (métodos clássicos):** NumPy, OpenCV, scikit-image e scikit-learn, em CPU.
  - Pré-processamento: equalização de histograma e filtro gaussiano.
  - Descritores: HOG, Gabor, LBP e DCT.
  - Classificadores: SVM linear, SVM RBF (com PCA) e k-NN.
- **Comparação justa:** as duas frentes usam os mesmos dados, a mesma limpeza, a mesma divisão, as mesmas degradações e as mesmas métricas, todos vindos de `src/common/`. Os hiperparâmetros são escolhidos na validação.
- **Regra do teste:** o teste é usado **uma única vez por método**, na semana 5, numa execução que avalia o teste limpo e os 9 níveis de degradação. Durante o desenvolvimento, as degradações rodam na validação.
- **Degradações:** ruído gaussiano (σ 5, 10, 20), desfoque gaussiano (núcleo 3, 5, 7) e redução de resolução (48 → 24 → 12).
- **Métricas:** acurácia, F1 por classe e macro, matriz de confusão, tempo de treino e de inferência por imagem.

### Cronograma

| Semana | Período | Frente A (Antonio e Kauã) | Frente B (Yago e Luccas) |
|---|---|---|---|
| 1 | 06–12/11 | ✅ Ambiente PyTorch na RTX 5060 | ✅ Limpeza da FER-2013, `splits/`, `src/common/` |
| 2 | 13–19/11 | ✅ CNN do zero e ResNet-18 na FER-2013 | 🟡 Pré-processamento e descritores (HOG pronto) |
| 3 | 20–26/11 | ✅ Demo na webcam (vídeo no CK+ virou extra) | 🟡 SVM linear e linhas de base (k-NN pendente) |
| 4 | 27/11–03/12 | Balanceamento, Grad-CAM, FFT2, robustez e demo ao vivo | Gabor, LBP, DCT, k-NN, SVM RBF, robustez e tabela comparativa |
| 5 | 04–10/12 | Teste final, análise conjunta, slides e ensaio da demo | Teste final, análise conjunta, slides e relatório |

## Estrutura do repositório

```
src/common/    base comum (Frente B): dados, limpeza, degradações, métricas
src/cnn/       Frente A: datasets, modelos, laço de treino, scripts de treino, demo na webcam
src/classic/   Frente B: pré-processamento, descritores e classificadores
scripts/       download dos dados, verificação do ambiente, figuras
splits/        decisão de limpeza por imagem e folds do CK+ (versionados)
results/       métricas de cada execução, em JSON (versionados)
reports/       relatórios, figuras e slides
tests/         pytest (common, cnn, classic); rodam em CPU, sem os dados reais
```

## Dados

As duas frentes usam **os mesmos dados, a mesma limpeza e a mesma divisão**, e todos vêm de `src/common/` (contrato na seção 5 do [CLAUDE.md](CLAUDE.md)). Os dados e os pesos nunca entram no git, mas os arquivos de divisão em `splits/` entram.

| Base | Fonte | Conteúdo | Divisão |
|---|---|---|---|
| FER-2013 | Competição oficial do Kaggle ([challenges-in-representation-learning-facial-expression-recognition-challenge](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)) | 35.887 imagens 48×48 em cinza, 7 classes | `Training` → treino (28.709), `PublicTest` → validação (3.589), `PrivateTest` → teste (3.589) |
| CK+ | Pedido aos autores (Lucey et al., 2010), **acesso restrito** | 593 sequências de 123 pessoas, das quais 327 têm rótulo | validação cruzada em 10 partes agrupadas por pessoa (`splits/ckplus_folds.csv`) |

### 1. Passos manuais (uma vez por pessoa)

Só uma pessoa pode fazer estes passos, e nenhum script faz por ela:

1. **Aceitar as regras da competição:** entrar no Kaggle, abrir <https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/rules> e clicar em *I Understand and Accept*. O Kaggle pode pedir verificação por telefone.
2. **Token da API:** em <https://www.kaggle.com/settings>, na seção *API*, clicar em *Create New Token* e mover o `kaggle.json` baixado para `~/.kaggle/kaggle.json` (no Windows, `C:\Users\<você>\.kaggle\kaggle.json`):
   ```powershell
   New-Item -ItemType Directory -Force $env:USERPROFILE\.kaggle
   Move-Item $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
   ```
   Uma alternativa é definir as variáveis `KAGGLE_USERNAME` e `KAGGLE_KEY`. **Nunca** cole o token em código, notebook, chat ou commit (`kaggle.json` está no `.gitignore`).

### 2. Preparar o ambiente e os dados

```bash
python -m venv .venv
.venv\Scripts\activate                  # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt         # a Frente A instala o PyTorch antes (ver "Ambiente da Frente A")

python scripts/download_data.py         # baixa, extrai e valida a FER-2013; prepara a pasta do CK+
python -m src.common.prepare_data       # cache, limpeza, splits, relatório e (se houver) o CK+
pytest                                  # testes (não precisam dos dados reais)
```

- **`scripts/download_data.py`** é idempotente: o que já existe e está válido não é baixado de novo.
  - Sem o token, ou com erro 401/403, o script **para** e explica os dois passos acima.
  - Ele extrai `fer2013.csv` do `fer2013.tar.gz` que vem no zip da competição. Se o tar.gz faltar, usa `icml_face_data.csv`, que tem os mesmos dados, com os nomes das colunas normalizados.
  - Depois valida 35.887 linhas, 3 colunas, 2.304 pixels por linha e 28.709 / 3.589 / 3.589 por `Usage`, e grava o SHA-256 em `data/raw/fer2013/CHECKSUM.txt`.
- **`python -m src.common.prepare_data`** aceita as opções `--std-threshold <valor>` (limiar de imagem quase constante, 5 por padrão), `--skip-face-check` (pula o Viola-Jones de sinalização, que é mais lento) e `--force` (refaz os caches).

**Conferência entre máquinas.** O SHA-256 esperado do `fer2013.csv` (extraído do `fer2013.tar.gz` oficial, 301.072.766 bytes) é

```
3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b
```

O script compara o hash sozinho e avisa se ele for diferente. Também dá para conferir à mão com `certutil -hashfile data\raw\fer2013\fer2013.csv SHA256` (Windows) ou `sha256sum data/raw/fer2013/fer2013.csv` (Linux/macOS).

**Se a competição oficial der erro 403.** Algumas contas recebem 403 mesmo depois de aceitar as regras. Nesse caso, o dataset público [`deadskull7/fer2013`](https://www.kaggle.com/datasets/deadskull7/fer2013) tem **exatamente o mesmo arquivo**: o SHA-256 bate com o oficial (conferido pela Frente A em 22/11). Use-o só assim, conferindo o hash:

```powershell
.venv\Scripts\kaggle datasets download -d deadskull7/fer2013 -p data\raw\fer2013 --unzip
.venv\Scripts\python scripts\download_data.py      # valida e confere o SHA-256
.venv\Scripts\python -m src.common.prepare_data
```

> **Não use** as versões da FER-2013 em pastas de imagens (como a `msambare/fer2013` do Kaggle), porque elas perderam a divisão PublicTest/PrivateTest de que o projeto precisa. Qualquer fonte diferente das duas acima só vale se o SHA-256 bater.

### 3. CK+ (quando o acesso chegar)

O CK+ não tem download público oficial e **não deve ser baixado de espelhos** (Kaggle, GitHub, Google Drive de terceiros), por causa da licença.

1. Alguém da equipe pede o acesso aos autores pela página de recursos do laboratório do Prof. Jeffrey Cohn (Universidade de Pittsburgh).
2. Os zips chegam por e-mail. Coloque em `data/raw/ckplus/`:
   - `extended-cohn-kanade-images.zip` e `Emotion_labels.zip` (obrigatórios);
   - `FACS_labels.zip` e `Landmarks.zip` (opcionais).
3. Rode `python scripts/download_data.py --only ckplus` e depois `python -m src.common.prepare_data`.

Enquanto isso, `data/raw/ckplus/LEIA-ME.txt` repete essas instruções. Os scripts avisam que o CK+ falta e seguem sem erro.

Regras do CK+ no projeto:
- **Sequências:** só entram as que têm rótulo. O "desprezo" (rótulo 2) é descartado, e o resto é mapeado para a ordem da FER-2013.
- **Recorte do rosto:** o rosto é recortado com Viola-Jones no primeiro quadro (o maior rosto), e a mesma caixa vale para a sequência inteira. Depois, o quadro vai para cinza e é redimensionado para 48×48.
- **Classe neutro:** sai do primeiro quadro de cada sequência.

### 4. O que fica versionado

| Versionado | Fora do git |
|---|---|
| `splits/fer2013_clean.csv`: decisão de limpeza por imagem (`id, split, label, keep, reason`) | `data/raw/`: CSV da FER-2013, zips e pastas do CK+ |
| `splits/ckplus_folds.csv`: fold de cada sequência (`subject, sequence, label, fold`) | `data/processed/`: caches `.npz` e a grade de conferência da limpeza |
| [`reports/data_cleaning.md`](reports/data_cleaning.md): o que foi removido, por motivo e por classe | `kaggle.json`, `.venv/`, `checkpoints/`, `*.pt` |
| `results/*.json`: métricas no formato do contrato | `tests/fixtures/ckplus_fake/` (gerada pelos testes) |

Os pesos treinados ficam no Google Drive da equipe: (link a adicionar).

### 5. Limpeza da FER-2013

Nenhuma imagem é apagada. A coluna `keep` decide o que entra, e **só o treino perde imagens**. A validação e o teste ficam intactos, para os números continuarem comparáveis com a literatura, e neles os problemas são só sinalizados com `flag:...`.

| Motivo | O que acontece |
|---|---|
| Duplicata exata (MD5 dos pixels) dentro do treino | Fica só a cópia de menor `id` |
| Cópia no treino de uma imagem da validação ou do teste (vazamento) | Sai do treino |
| Cópias idênticas com rótulos diferentes | Saem do treino |
| Imagem quase constante (desvio padrão < 5) | Sai do treino |
| Viola-Jones sem rosto | Só sinaliza, porque falha muito em 48×48 |

Resultado: o treino passou de 28.709 para **26.911 imagens** (1.798 removidas, 703 delas por vazamento para a validação ou o teste). Os números por motivo e classe estão em [`reports/data_cleaning.md`](reports/data_cleaning.md).

### 6. Como usar nos códigos das frentes

```python
from src.common.data import load_fer2013, load_ckplus, EMOTIONS
from src.common.clean import class_weights
from src.common.degrade import apply_degradation, LEVELS
from src.common.metrics import evaluate, save_results
from src.common.utils import set_seed

set_seed(42)
X_tr, y_tr, ids_tr = load_fer2013("train")              # uint8 [N,48,48], int64 [N], id = linha do CSV
X_val, y_val, _ = load_fer2013("val")                   # o teste ("test") só na avaliação final
w = class_weights(y_tr)                                 # [7], esquema "balanced" do scikit-learn
X_noisy = apply_degradation(X_val, "noise", 10)         # LEVELS lista os níveis de cada degradação
Xc, yc, idc = load_ckplus(fold=0, part="train", mode="image")        # pico + neutro
seqs, ys, ids = load_ckplus(fold=0, part="test", mode="sequence")    # lista de [T,48,48]

meta = {"method": "hog_linearsvc", "track": "classic", "dataset": "fer2013", "split": "val",
        "degradation": "noise", "level": 10, "train_time_s": 12.3, "inference_ms_per_img": 0.4, "seed": 42}
save_results(evaluate(y_val, y_pred, meta))             # grava results/hog_linearsvc_fer2013_val_noise_10.json
```

## Ambiente da Frente A (redes neurais)

A RTX 5060 é da série RTX 50 (arquitetura Blackwell, `sm_120`) e **exige PyTorch compilado com CUDA 12.8 ou mais novo**. Instale o PyTorch antes do resto:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

A versão deve terminar em `+cu128` (ou um CUDA mais novo) e `is_available()` deve dar `True`. Se aparecer **`RuntimeError: CUDA error: no kernel image is available for execution on the device`**, o PyTorch instalado não tem kernels para a Blackwell (build antigo ou `+cpu`). Desinstale e reinstale com o comando acima.

O `requirements.txt` fixa o OpenCV em 4.14, porque o OpenCV 5 removeu o `cv2.CascadeClassifier` (Viola-Jones).

Verificação e preparação:

```bash
python scripts/check_env.py          # versões, GPU, autocast e benchmark da ResNet-18
python scripts/cache_pretrained.py   # pesos ImageNet da ResNet-18 no cache local (treino sem internet)
```

Resultados do `check_env.py` na RTX 5060 (ResNet-18, lote 64, 224×224, forward + backward):

| Precisão | ms/passo | imagens/s | pico de memória (MiB) |
|---|---|---|---|
| fp32 | _pendente_ | _pendente_ | _pendente_ |
| AMP bf16 | _pendente_ | _pendente_ | _pendente_ |

Versões: _pendente_ (Python, PyTorch, CUDA, driver).

**Reserva:** [`notebooks/00_colab_setup.ipynb`](notebooks/00_colab_setup.ipynb) roda o mesmo fluxo no Colab (GPU T4), com checkpoints e cache de pesos no Google Drive. As credenciais do Kaggle entram pelos *Secrets* do Colab, nunca no notebook.

## Como rodar

### Frente A: treinos e demo

```powershell
# Treinos (cerca de 6 min cada na RTX 5060); gravam checkpoints/ e results/
.venv\Scripts\python -m src.cnn.train_scratch
.venv\Scripts\python -m src.cnn.train_resnet

# Execução rápida para conferir se tudo roda (2 lotes por época)
.venv\Scripts\python -m src.cnn.train_scratch --epochs 1 --limit-batches 2 --results-dir $env:TEMP\r --ckpt-dir $env:TEMP\c

# Demo na webcam (tecla q para sair)
.venv\Scripts\python -m src.cnn.webcam_demo                      # usa checkpoints\resnet18.pt
.venv\Scripts\python -m src.cnn.webcam_demo --checkpoint checkpoints\cnn_scratch.pt --camera 1
```

Opções, decisões e resultados detalhados: [`reports/frente_a_modelos.md`](reports/frente_a_modelos.md).

### Frente B: métodos clássicos

```bash
python -m src.classic.train_classic                       # HOG + SVM linear, C escolhido na validação
python -m src.classic.train_classic --features pixels     # linha de base: pixels brutos + SVM linear
python -m src.classic.train_classic --model majority      # linha de base: classe majoritária
```

Os scripts de treino nunca carregam o teste. Os hiperparâmetros são escolhidos na validação.

### Demo ao vivo (apresentação final)

Em preparação na semana 4 (Antonio): suavização das previsões, barras com as 7 probabilidades, tela cheia para o projetor, Grad-CAM do quadro congelado e vídeo gravado como plano B. A demo não grava nenhuma imagem em disco. O comando, as teclas e o checklist do dia vão entrar aqui.

## Resultados

**Resultados preliminares, na validação** (PublicTest, 3.589 imagens). O teste ainda não foi usado.

| Método | Frente | Acurácia | F1 macro | Treino | Inferência |
|---|---|---|---|---|---|
| **ResNet-18 (ImageNet, 224×224)** | A | **0,671** | **0,626** | 6,0 min (GPU) | 0,31 ms/img (GPU) |
| CNN do zero (48×48) | A | 0,633 | 0,528 | 5,7 min (GPU) | 0,05 ms/img (GPU) |
| HOG + SVM linear | B | 0,405 | 0,332 | 3,5 min (CPU) | 1,22 ms/img (CPU) |
| Pixels + SVM linear | B | 0,352 | 0,255 | 17,7 min (CPU) | 0,06 ms/img (CPU) |
| Classe majoritária (referência mínima) | B | 0,249 | 0,057 | — | — |

F1 por classe:

| Classe | CNN do zero | ResNet-18 |
|---|---|---|
| raiva | 0,56 | 0,59 |
| nojo | 0,00 | 0,45 |
| medo | 0,37 | 0,50 |
| alegria | 0,85 | 0,87 |
| tristeza | 0,54 | 0,55 |
| surpresa | 0,79 | 0,79 |
| neutro | 0,58 | 0,63 |

Os números completos de cada execução estão em [`results/`](results/).

## Discussão

Observações preliminares, ainda sem a avaliação final:
- As redes neurais ficam mais de 25 pontos de acurácia acima dos métodos clássicos testados até agora. Na literatura, uma VGG ajustada com cuidado chega a 73,3% no teste da FER-2013 (Khaireddin; Chen, 2021).
- A transferência do ImageNet ajuda mesmo em imagens cinza de 48×48: a ResNet-18 ganha 3,8 pontos de acurácia e 0,10 de F1 macro sobre a CNN do zero, com menos épocas.
- **Nojo** é a classe mais difícil: é a mais rara da base, a CNN do zero nunca a prevê e a ResNet-18 acerta 18 de 56. O balanceamento de classes é o próximo experimento.
- As confusões principais (medo ↔ tristeza, tristeza ↔ neutro, raiva ↔ tristeza) são as esperadas na literatura.
- O HOG melhora cerca de 5 pontos sobre os pixels brutos com o mesmo classificador, o que mostra o ganho de um descritor de bordas.

## Conclusão
(em desenvolvimento)

## Referências

AHMED, N.; NATARAJAN, T.; RAO, K. R. Discrete cosine transform. **IEEE Transactions on Computers**, v. C-23, n. 1, p. 90–93, 1974.

DALAL, N.; TRIGGS, B. Histograms of oriented gradients for human detection. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2005. **Proceedings** [...]. p. 886–893.

EKMAN, P.; FRIESEN, W. V. Constants across cultures in the face and emotion. **Journal of Personality and Social Psychology**, v. 17, n. 2, p. 124–129, 1971.

GONZALEZ, R. C.; WOODS, R. E. **Processamento digital de imagens**. 3. ed. São Paulo: Pearson Prentice Hall, 2010.

GOODFELLOW, I. J. et al. Challenges in representation learning: a report on three machine learning contests. **Neural Networks**, v. 64, p. 59–63, 2015.

HE, K. et al. Deep residual learning for image recognition. In: IEEE CONFERENCE ON COMPUTER VISION AND PATTERN RECOGNITION (CVPR), 2016. **Proceedings** [...]. p. 770–778.

KHAIREDDIN, Y.; CHEN, Z. Facial emotion recognition: state of the art performance on FER2013. **arXiv preprint** arXiv:2105.03588, 2021.

LI, S.; DENG, W. Deep facial expression recognition: a survey. **IEEE Transactions on Affective Computing**, v. 13, n. 3, p. 1195–1215, 2022.

LUCEY, P. et al. The Extended Cohn-Kanade Dataset (CK+): a complete dataset for action unit and emotion-specified expression. In: IEEE CVPR WORKSHOPS, 2010. **Proceedings** [...]. p. 94–101.

LYONS, M. et al. Coding facial expressions with Gabor wavelets. In: IEEE INTERNATIONAL CONFERENCE ON AUTOMATIC FACE AND GESTURE RECOGNITION, 3., 1998. **Proceedings** [...]. p. 200–205.

SELVARAJU, R. R. et al. Grad-CAM: visual explanations from deep networks via gradient-based localization. In: IEEE INTERNATIONAL CONFERENCE ON COMPUTER VISION (ICCV), 2017. **Proceedings** [...]. p. 618–626.

SHAN, C.; GONG, S.; McOWAN, P. W. Facial expression recognition based on Local Binary Patterns: a comprehensive study. **Image and Vision Computing**, v. 27, n. 6, p. 803–816, 2009.

SIMONYAN, K.; ZISSERMAN, A. Very deep convolutional networks for large-scale image recognition. In: INTERNATIONAL CONFERENCE ON LEARNING REPRESENTATIONS (ICLR), 2015.

VIOLA, P.; JONES, M. J. Robust real-time face detection. **International Journal of Computer Vision**, v. 57, n. 2, p. 137–154, 2004.
