# Limpeza da FER-2013

> Arquivo gerado por `python -m src.common.prepare_data` em
> 2026-11-13 08:18 UTC. Não edite à mão: rode o comando de novo.

- **Arquivo de origem:** `data/raw/fer2013/fer2013.csv`, SHA-256 `3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b`
- **Decisão por imagem:** [`splits/fer2013_clean.csv`](../splits/fer2013_clean.csv) (colunas `id, split, label, keep, reason`)
- **Limiar de imagem quase constante:** desvio padrão dos pixels < **5**
- **Sinalização por Viola-Jones:** sim
- **Grupos de imagens idênticas (MD5 dos pixels):** 1516

**Regras.** Só o treino perde imagens. A validação (PublicTest) e o teste (PrivateTest) ficam
intactos, para os números continuarem comparáveis com a literatura. Neles, os problemas
são só sinalizados (`flag:...`). Nenhum arquivo é apagado: quem decide é a coluna `keep`.

| motivo | o que é | treino | val / teste |
|---|---|---|---|
| `dup_within_split` | cópia exata de outra imagem do treino com o mesmo rótulo | sai (fica a de menor `id`) | `flag:dup_within_split` |
| `dup_cross_split_leak` | a imagem do treino também está em val ou teste (vazamento) | sai | intacta |
| `label_conflict` | cópias idênticas com rótulos diferentes | todas as cópias saem | `flag:label_conflict` |
| `low_std` | imagem quase constante (vazia, preta, branca) | sai | `flag:low_std` |
| `flag:dup_val_test` | a mesma imagem em val e em teste | não se aplica | só sinaliza |
| `flag:no_face` | o Viola-Jones não achou rosto (ampliada para 96x96) | só sinaliza | só sinaliza |

## Resumo por conjunto

| conjunto | antes | removidas | depois | sinalizadas (ficam) |
|---|---:|---:|---:|---:|
| train | 28709 | 1798 | 26911 | 11062 |
| val | 3589 | 0 | 3589 | 1509 |
| test | 3589 | 0 | 3589 | 1510 |

## Imagens removidas do treino, por motivo e classe

Uma imagem pode ter mais de um motivo. A última linha conta cada imagem uma vez só.

| motivo | 0 raiva | 1 nojo | 2 medo | 3 alegria | 4 tristeza | 5 surpresa | 6 neutro | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dup_within_split | 88 | 35 | 173 | 126 | 90 | 421 | 88 | 1021 |
| dup_cross_split_leak | 116 | 47 | 109 | 58 | 50 | 275 | 48 | 703 |
| label_conflict | 27 | 4 | 33 | 3 | 21 | 28 | 11 | 127 |
| low_std | 7 | 0 | 0 | 1 | 1 | 1 | 2 | 12 |
| imagens removidas (únicas) | 219 | 86 | 300 | 185 | 153 | 711 | 144 | 1798 |

## Sinalizações (as imagens ficam), por conjunto

| sinalização | train | val | test |
|---|---:|---:|---:|
| flag:dup_within_split | 0 | 52 | 33 |
| flag:dup_val_test | 0 | 46 | 44 |
| flag:label_conflict | 0 | 16 | 17 |
| flag:low_std | 0 | 2 | 0 |
| flag:no_face | 11431 | 1422 | 1439 |

O Viola-Jones sinalizou 14292 imagens no total. Em 48x48 o detector falha muito,
até em rostos bons, então esse número **não** mede a qualidade da base e nunca é usado para
remover imagens.

## Distribuição de classes antes e depois

| classe | treino antes | treino depois | val | teste |
|---|---:|---:|---:|---:|
| 0 raiva | 3995 (13.9%) | 3776 (14.0%) | 467 (13.0%) | 491 (13.7%) |
| 1 nojo | 436 (1.5%) | 350 (1.3%) | 56 (1.6%) | 55 (1.5%) |
| 2 medo | 4097 (14.3%) | 3797 (14.1%) | 496 (13.8%) | 528 (14.7%) |
| 3 alegria | 7215 (25.1%) | 7030 (26.1%) | 895 (24.9%) | 879 (24.5%) |
| 4 tristeza | 4830 (16.8%) | 4677 (17.4%) | 653 (18.2%) | 594 (16.6%) |
| 5 surpresa | 3171 (11.0%) | 2460 (9.1%) | 415 (11.6%) | 416 (11.6%) |
| 6 neutro | 4965 (17.3%) | 4821 (17.9%) | 607 (16.9%) | 626 (17.4%) |

## Pesos de classe do treino limpo

`src.common.clean.class_weights(y)`: inverso da frequência normalizado,
`N / (7 · n_c)`, igual ao `class_weight="balanced"` do scikit-learn. A classe com
menos exemplos (nojo) recebe o maior peso.

| classe | peso |
|---|---:|
| 0 raiva | 1.018 |
| 1 nojo | 10.984 |
| 2 medo | 1.012 |
| 3 alegria | 0.547 |
| 4 tristeza | 0.822 |
| 5 surpresa | 1.563 |
| 6 neutro | 0.797 |

## Conferência visual do limiar

A grade com as imagens marcadas e as que estão logo acima do limiar, e o histograma
do desvio padrão, ficam em `data/processed/inspection/fer2013_low_std_grid.png` e no `_hist.png` ao lado. Esses arquivos
não são versionados porque contêm imagens da base. Para trocar o limiar:
`python -m src.common.prepare_data --std-threshold <valor>`.
