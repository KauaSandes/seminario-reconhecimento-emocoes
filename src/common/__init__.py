"""Base comum das duas frentes (CLAUDE.md, seção 5).

Toda leitura de dados, degradação e métrica usada na comparação sai daqui:

- ``src.common.data``: ``load_fer2013(split)`` e ``load_ckplus(fold, part, mode)``.
- ``src.common.clean``: limpeza da FER-2013 e ``class_weights(y)``.
- ``src.common.degrade``: ruído, desfoque e redução de resolução (só no teste).
- ``src.common.metrics``: ``evaluate(...)`` e ``save_results(...)`` no formato do JSON.

Mudanças aqui afetam as duas frentes e precisam de revisão de alguém de cada uma.
"""
