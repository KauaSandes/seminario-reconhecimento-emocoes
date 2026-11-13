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
pip install -r requirements.txt         # a Frente A instala o PyTorch antes (ver abaixo)

python scripts/download_data.py         # baixa, extrai e valida a FER-2013; prepara a pasta do CK+
python -m src.common.prepare_data       # cache, limpeza, splits, relatório e (se houver) o CK+
pytest                                  # testes da base comum (não precisam dos dados reais)
```

- **`scripts/download_data.py`** é idempotente: o que já existe e está válido não é baixado de novo.
  - Sem o token, ou com erro 401/403, o script **para** e explica os dois passos acima.
  - Ele extrai `fer2013.csv` do `fer2013.tar.gz` que vem no zip da competição. Se o tar.gz faltar, usa `icml_face_data.csv`, que tem os mesmos dados, com os nomes das colunas normalizados.
  - Depois valida 35.887 linhas, 3 colunas, 2.304 pixels por linha e 28.709 / 3.589 / 3.589 por `Usage`, e grava o SHA-256 em `data/raw/fer2013/CHECKSUM.txt`.
- **`python -m src.common.prepare_data`** aceita as opções `--std-threshold <valor>` (limiar de imagem quase constante, 5 por padrão), `--skip-face-check` (pula o Viola-Jones de sinalização, que é mais lento) e `--force` (refaz os caches).

**Conferência entre máquinas.** O SHA-256 esperado do `fer2013.csv` é `pendente (preenchido após o primeiro download)`. O script compara o hash sozinho e avisa se ele for diferente.

> **Não use** as versões da FER-2013 em pastas de imagens (como a `msambare/fer2013` do Kaggle), porque elas perderam a divisão PublicTest/PrivateTest de que o projeto precisa. Se o download oficial falhar, avise a equipe em vez de trocar de fonte.

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

### 5. Limpeza da FER-2013

Nenhuma imagem é apagada. A coluna `keep` decide o que entra, e **só o treino perde imagens**. A validação e o teste ficam intactos, para os números continuarem comparáveis com a literatura, e neles os problemas são só sinalizados com `flag:...`.

| Motivo | O que acontece |
|---|---|
| Duplicata exata (MD5 dos pixels) dentro do treino | Fica só a cópia de menor `id` |
| Cópia no treino de uma imagem da validação ou do teste (vazamento) | Sai do treino |
| Cópias idênticas com rótulos diferentes | Saem do treino |
| Imagem quase constante (desvio padrão < 5) | Sai do treino |
| Viola-Jones sem rosto | Só sinaliza, porque falha muito em 48×48 |

Os números estão em [`reports/data_cleaning.md`](reports/data_cleaning.md).

### 6. Como usar nos códigos das frentes

```python
from src.common.data import load_fer2013, load_ckplus, EMOTIONS
from src.common.clean import class_weights
from src.common.degrade import apply_degradation, LEVELS
from src.common.metrics import evaluate, save_results
from src.common.utils import set_seed

set_seed(42)
X_tr, y_tr, ids_tr = load_fer2013("train")              # uint8 [N,48,48], int64 [N], id = linha do CSV
X_te, y_te, _ = load_fer2013("test")
w = class_weights(y_tr)                                 # [7], esquema "balanced" do scikit-learn
X_noisy = apply_degradation(X_te, "noise", 10)          # só no teste; LEVELS lista os níveis
Xc, yc, idc = load_ckplus(fold=0, part="train", mode="image")        # pico + neutro
seqs, ys, ids = load_ckplus(fold=0, part="test", mode="sequence")    # lista de [T,48,48]

meta = {"method": "hog_linearsvc", "track": "classic", "dataset": "fer2013", "split": "test",
        "degradation": "noise", "level": 10, "train_time_s": 12.3, "inference_ms_per_img": 0.4, "seed": 42}
save_results(evaluate(y_te, y_pred, meta))              # grava results/hog_linearsvc_fer2013_test_noise_10.json
```

**Ajustes na Frente A (branch `cnn/data-pipeline`).** Os stubs de `tests/conftest.py` supunham uma API um pouco diferente:
- `load_fer2013` e `load_ckplus` devolvem **três** valores `(X, y, ids)`. Basta trocar `images, labels = ...` por `images, labels, _ = ...` em `src/cnn/datasets.py` e `src/cnn/video_data.py`.
- As métricas são gravadas com `save_results(evaluate(y_true, y_pred, meta))`, no lugar do `save_results(**kwargs)` do stub.
- Para o ruído sair igual nas duas frentes, usem `src.common.degrade.apply_degradation(X, kind, level, seed)` em lote. Os nomes `"noise"`, `"blur"` e `"resolution"` são os mesmos que vocês já usam.

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
