# Frente A: CNN do zero, ResNet-18 e demo na webcam

Relatório da etapa entregue no PR #2 (branch `cnn/modelos-fer2013`), em 22/11/2026. Ele explica o que foi feito, como rodar, o que os números querem dizer e o que falta, para quem for continuar.

## 1. Resumo

| Modelo | Acurácia (val) | F1 macro (val) | Épocas (melhor) | Treino | Inferência |
|---|---|---|---|---|---|
| CNN do zero, 48×48 | 0,633 | 0,528 | 57 (49) | 5,7 min | 0,05 ms/img |
| **ResNet-18 ImageNet, 224×224** | **0,671** | **0,626** | 14 (9) | 6,0 min | 0,31 ms/img |

- A **ResNet-18 é o melhor modelo** até agora, e é o padrão da demo na webcam.
- Os números são da **validação** (PublicTest, 3.589 imagens). **O split de teste não foi tocado**: pelo contrato (CLAUDE.md, seção 5), ele é usado uma única vez por método, no final.
- Para comparar: Khaireddin e Chen (2021) chegam a 73,3% no teste da FER-2013 com uma VGG ajustada com cuidado.
- Tudo rodou na RTX 5060 (8 GB), com precisão mista e seed 42. A suíte de testes passou de 117 para **130 testes**, todos passando.

## 2. Escopo desta etapa

- **Semana 2 da frente A**, que estava atrasada: treinar a CNN do zero e a ResNet-18 na FER-2013.
- **Demo na webcam**, da semana 3.
- **O vídeo no CK+ (CNN + LSTM) passou a ser extra**, por decisão da frente A. Além disso, o acesso ao CK+ ainda não chegou. O `src/cnn/video_data.py` do Antonio continua no repositório, se der tempo de voltar a ele.
- `src/common/` **não foi alterado**.

## 3. Dados nesta máquina

O download pela competição oficial do Kaggle foi recusado (403) na conta usada. O `fer2013.csv` foi baixado então do dataset público [`deadskull7/fer2013`](https://www.kaggle.com/datasets/deadskull7/fer2013), que não depende de aceitar as regras da competição.

**Ele é idêntico ao oficial.** O SHA-256 (`3b8d9617…968f871b`) bate com o `EXPECTED_SHA256` que o Yago fixou em `scripts/download_data.py`. Depois, o fluxo normal (`download_data.py` e `prepare_data`) gerou um `splits/fer2013_clean.csv` idêntico ao do repositório. **Se o download oficial falhar para mais alguém**, este é o caminho:

```powershell
.venv\Scripts\kaggle datasets download -d deadskull7/fer2013 -p data\raw\fer2013 --unzip
.venv\Scripts\python scripts\download_data.py      # valida e confere o SHA-256
.venv\Scripts\python -m src.common.prepare_data
```

## 4. O que foi implementado

Todo o código novo está em `src/cnn/`. Ele usa os loaders de `src/cnn/datasets.py` (Antonio) e as métricas de `src/common/metrics.py` (Yago), sem reimplementar nada.

| Arquivo | Conteúdo |
|---|---|
| `models.py` | `SmallCNN`, `build_resnet18`, `set_backbone_trainable`, `build_model(name)`, `save_checkpoint`, `load_checkpoint`. O dicionário `MODELS = {"cnn_scratch": "small", "resnet18": "resnet"}` liga cada modelo ao modo de entrada do `datasets.py`. |
| `engine.py` | Laço de treino compartilhado: `train_one_epoch`, `predict`, `fit` (parada antecipada), além de `make_parser`, `setup` e `finish`, usados pelos dois scripts para não duplicar argparse, loaders, checkpoint e JSON. |
| `train_scratch.py` | Treino da CNN do zero. |
| `train_resnet.py` | Treino da ResNet-18 em duas fases. |
| `webcam_demo.py` | Demo: Viola-Jones mais o modelo treinado. |
| `tests/cnn/test_models.py`, `test_engine.py`, `test_webcam_demo.py` | 13 testes novos, em CPU, com a fixture `fake_data` (dados falsos, sem baixar pesos). |

### 4.1 Modelos

- **`SmallCNN`** (3,5 M de parâmetros): 3 blocos, cada um com 2 × [conv 3×3 + BatchNorm + ReLU] + MaxPool, com 64, 128 e 256 canais. A entrada 1×48×48 vira 256×6×6. A cabeça é Dropout(0,5) → Linear(9216, 256) → ReLU → Dropout(0,5) → Linear(256, 7).
- **ResNet-18** (11,2 M de parâmetros): pesos `IMAGENET1K_V1` do torchvision, com a `fc` trocada por `Linear(512, 7)`. A entrada é a imagem em cinza redimensionada para 224×224, repetida nos 3 canais e com a normalização do ImageNet. Isso tudo é feito pelo `datasets.py`, no modo `"resnet"`.

### 4.2 Treino

- Adam, com **parada antecipada pela perda de validação**: o melhor estado fica guardado e é restaurado no fim. Se a perda de validação não for finita em nenhuma época (treino divergiu), o script para com erro em vez de gravar um checkpoint ruim.
- Precisão mista (autocast fp16 + GradScaler) em CUDA; `--no-amp` desliga. Seed 42 via `src.common.utils.set_seed`.
- **ResNet-18 em duas fases:**
  - Fase 1: só a `fc` treina, por 3 épocas com lr 1e-3. O resto da rede fica com `requires_grad=False`, mas as estatísticas da BatchNorm continuam se ajustando à FER, o que funciona como uma adaptação de domínio.
  - Fase 2: a rede inteira treina com lr 1e-4 e parada antecipada (paciência 5). As épocas são numeradas em sequência depois da fase 1.
- Os scripts sempre chamam `make_loaders(..., splits=("train", "val"))`. **Não há caminho que carregue o teste.**
- No fim, o script grava o checkpoint em `checkpoints/{método}.pt`, avalia o melhor modelo na validação e grava o JSON do contrato em `results/{método}_fer2013_val_none_0.json`.

O checkpoint é um dicionário com as chaves `name`, `mode`, `size`, `state_dict`, `epoch`, `val_loss`, `history` (perda de treino, perda de validação e acurácia de validação por época) e `args`. O `load_checkpoint(path, device)` reconstrói o modelo sem baixar pesos.

### 4.3 Demo na webcam

Para cada quadro, a demo faz:
1. Converte para cinza e detecta os rostos com `src.common.utils.detect_faces` (Viola-Jones).
2. Recorta cada rosto com 10% de margem (`MARGIN = 0.1`) e redimensiona para 48×48.
3. Aplica **a mesma transformação da validação**, `build_transform(mode, augment=False, size)`, para casar com o treino.
4. Roda o modelo, com todos os rostos num único lote.
5. Desenha a caixa e o rótulo, por exemplo "alegria 87%".

## 5. Como rodar

Ambiente e dados: README, seções Ambiente e Dados (PyTorch com CUDA 12.8 ou mais novo na RTX 50).

```powershell
# Testes (CPU, cerca de 35 s)
.venv\Scripts\python -m pytest -q

# Treinos (cerca de 6 min cada na RTX 5060)
.venv\Scripts\python -m src.cnn.train_scratch
.venv\Scripts\python -m src.cnn.train_resnet

# Execução rápida para conferir se tudo roda (2 lotes por época)
.venv\Scripts\python -m src.cnn.train_scratch --epochs 1 --limit-batches 2 --results-dir $env:TEMP\r --ckpt-dir $env:TEMP\c

# Demo na webcam (tecla q para sair)
.venv\Scripts\python -m src.cnn.webcam_demo                      # usa checkpoints\resnet18.pt
.venv\Scripts\python -m src.cnn.webcam_demo --checkpoint checkpoints\cnn_scratch.pt --camera 1
```

Opções principais dos dois scripts de treino:

| Opção | Padrão |
|---|---|
| `--epochs` | 60 (CNN do zero) / 30 (ResNet-18) |
| `--patience` | 8 / 5 |
| `--lr` | 1e-3 / 1e-4 |
| `--batch-size` | 64 |
| `--balance {none,sampler,loss}` | none |
| `--workers` | 4 |
| `--seed` | 42 |
| `--device` | cuda |
| `--no-amp` | AMP ligado |
| `--ckpt-dir` | checkpoints/ |
| `--results-dir` | results/ |
| `--limit-batches` | todos os lotes |

Só no `train_resnet`:

| Opção | Padrão |
|---|---|
| `--size {224,112}` | 224 |
| `--head-epochs` | 3 |
| `--lr-head` | 1e-3 |
| `--no-pretrained` | só para testes |

**Pesos:** `checkpoints/` está no `.gitignore`. Os dois `.pt` desta etapa (`cnn_scratch.pt` com 14 MB e `resnet18.pt` com 45 MB) estão por enquanto só na máquina do Kauã e precisam ir para o Google Drive da equipe. Quem precisar deles antes disso pode retreinar com os comandos acima: com a seed fixa, o resultado deve sair igual ou muito próximo.

## 6. Resultados em detalhe (validação)

### 6.1 F1 por classe

| Classe | CNN do zero | ResNet-18 |
|---|---|---|
| raiva | 0,56 | 0,59 |
| **nojo** | **0,00** | **0,45** |
| medo | 0,37 | 0,50 |
| alegria | 0,85 | 0,87 |
| tristeza | 0,54 | 0,55 |
| surpresa | 0,79 | 0,79 |
| neutro | 0,58 | 0,63 |

### 6.2 Matriz de confusão da ResNet-18

Linhas são a classe verdadeira e colunas a prevista.

| | raiva | nojo | medo | alegria | tristeza | surpresa | neutro |
|---|---|---|---|---|---|---|---|
| **raiva** | **279** | 3 | 46 | 20 | 57 | 10 | 52 |
| **nojo** | 24 | **18** | 7 | 1 | 4 | 0 | 2 |
| **medo** | 57 | 0 | **232** | 15 | 91 | 47 | 54 |
| **alegria** | 18 | 0 | 10 | **788** | 18 | 20 | 41 |
| **tristeza** | 55 | 2 | 90 | 24 | **344** | 11 | 127 |
| **surpresa** | 17 | 0 | 24 | 16 | 6 | **335** | 17 |
| **neutro** | 26 | 1 | 27 | 55 | 78 | 8 | **412** |

### 6.3 Leitura

- **Alegria e surpresa são fáceis** para os dois modelos (F1 de 0,79 a 0,87). São expressões com mudanças grandes na boca e nos olhos.
- **Nojo é o problema.** Ele tem só 56 imagens na validação e é a classe mais rara no treino, com peso de classe 10,98. A CNN do zero **nunca prevê nojo** e manda 40 das 56 imagens para raiva. A ResNet-18 acerta 18 delas.
- **As confusões principais são as esperadas na literatura:** medo ↔ tristeza, tristeza ↔ neutro e raiva ↔ tristeza. São expressões sutis e parecidas em 48×48.
- **A transferência do ImageNet ajuda mesmo em imagens cinza de 48×48**: +3,8 pontos de acurácia e +0,10 de F1 macro, e com menos épocas.
- A acurácia máxima de validação durante o treino foi um pouco maior que a do checkpoint escolhido (0,645 contra 0,633 na CNN do zero, e 0,675 contra 0,671 na ResNet-18). Isso acontece porque o checkpoint é escolhido pela **perda** de validação, como pede o CLAUDE.md, e não pela acurácia.

## 7. Decisões e por quê

| Decisão | Motivo |
|---|---|
| Um `engine.py` compartilhado | Os dois scripts têm o mesmo laço, a mesma parada antecipada e o mesmo fechamento (checkpoint e JSON). Duplicar faria os dois divergirem. |
| Checkpoint escolhido pela perda de validação | É o que o CLAUDE.md pede para a parada antecipada. |
| `--balance none` por padrão | É o ponto de partida mais simples e comparável com a literatura. O balanceamento fica para o próximo experimento (seção 8). |
| ResNet a 224×224 | O tempo coube folgado (cerca de 25 s por época, 900 img/s com 4 workers, sem gargalo no DataLoader). Não precisamos cair para 112. |
| A demo usa `build_transform` do `datasets.py` | Garante que a imagem da webcam passa exatamente pelo mesmo pré-processamento da validação. |

## 8. Limitações e pendências

1. **A demo não foi testada com uma câmera de verdade.** Ela foi testada com imagens de validação coladas num quadro sintético, e funcionou de ponta a ponta. Falta alguém rodar na frente da webcam.
2. **Os pesos estão só numa máquina.** É preciso subir para o Drive (seção 5).
3. **O `git_commit` dos dois JSONs aponta para `282ccc4`** (a `main` antes desta etapa), porque os treinos rodaram com o código ainda não commitado. O código que gerou os números é o do commit `3d64796`. Quando o PR entrar, vale retreinar para o JSON registrar o commit certo. Leva cerca de 12 minutos no total.
4. **O balanceamento de classes ainda não foi testado.** `--balance loss` (perda ponderada) ou `--balance sampler` devem melhorar o F1 de nojo e o F1 macro, talvez com um pouco menos de acurácia. Com `--balance loss`, a perda de validação também fica ponderada, então compare as execuções por acurácia e F1, não pela perda.
5. **Não há agendador de taxa de aprendizado** (por exemplo `ReduceLROnPlateau` ou cosseno). Isso pode render alguns pontos.
6. **O `make_loaders` carrega o teste por padrão** quando `splits` não é passado (`splits=SPLITS`). Os scripts novos sempre passam `splits=("train", "val")`, mas vale mudar esse padrão num PR separado para evitar acidentes.

## 9. Próximos passos (semanas 4 e 5)

| Tarefa | Onde começar |
|---|---|
| **Grad-CAM** | `pytorch_grad_cam.GradCAM` (já testado no ambiente). Camada-alvo: `model.layer4[-1]` na ResNet-18, e a segunda conv do terceiro bloco na `SmallCNN` (`model.features[2][3]`). Carregar com `load_checkpoint`, usar imagens da **validação** e salvar as figuras em `reports/figures/`. |
| **FFT2 dos filtros** | Pesos da primeira camada: `model.conv1.weight` na ResNet (64×3×7×7; como a entrada é cinza repetida, some os 3 canais) e a primeira conv da `SmallCNN`, `model.features[0][0].weight` (64×1×3×3). Um filtro 3×3 é pequeno demais para um espectro legível, então aplique zero-padding (por exemplo para 32×32) antes da `fft2`. Comparar com os filtros de Gabor da frente B. |
| **Robustez** | `make_loaders(..., degradation=..., level=...)` já aplica as degradações de `src/common/degrade.py`, **só no teste**. Isso esbarra na regra de usar o teste uma vez só: **a equipe precisa decidir** se a robustez entra na avaliação final da semana 5, ou se durante o desenvolvimento ela roda na validação (o que exigiria mudar o `datasets.py` ou o contrato). |
| **Balanceamento e agendador** | `--balance loss` / `sampler` e um agendador, comparando sempre na validação. |
| **Avaliação final (semana 5)** | Uma única avaliação no teste por método, sem e com degradações. Ainda falta escrever um script `evaluate.py` que carregue o checkpoint e chame `predict` e `save_results` com `split="test"`. |

## 10. Para a frente B

- Os JSONs da frente A seguem o mesmo contrato e passam em `validate_record`, então a tabela comparativa pode ler `results/*.json` direto.
- O `requirements.txt` fixa o OpenCV em 4.14 porque **o OpenCV 5 removeu o `cv2.CascadeClassifier`** (Viola-Jones).
- Os filtros de Gabor de vocês serão comparados com o espectro dos filtros aprendidos (seção 9). Se puderem salvar o banco de Gabor usado (ou os parâmetros: frequências e orientações), fica fácil montar a figura lado a lado.

## 11. Como esta etapa foi feita: pipeline com agentes

Esta etapa foi produzida no Claude Code com um pipeline de agentes que ficou no repositório para as próximas etapas:

- `.claude/agents/producao.md` (Sonnet): implementa código e testes a partir de uma especificação.
- `.claude/agents/validacao.md` (Sonnet): revisa contra a especificação e o contrato e roda execuções curtas com dados reais. Não edita código.
- `.claude/agents/teste.md` (Haiku): pytest completo, contrato dos JSONs, higiene do git e uso do split de teste.
- `.claude/skills/pipeline/SKILL.md`: o roteiro do modelo líder (Opus). Ele especifica, delega, decide sobre os achados, roda os treinos longos e faz o commit.

Nesta etapa, a validação encontrou um caso real (um treino que diverge seria salvo sem aviso), corrigido antes do merge. Para usar o pipeline, abra o Claude Code na raiz do repositório e peça para avançar uma etapa do cronograma; a skill `pipeline` descreve o fluxo.
