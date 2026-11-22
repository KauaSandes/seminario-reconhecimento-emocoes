---
name: pipeline
description: Executa uma etapa do seminário com o pipeline produção → validação → teste, com o modelo principal como líder e os subagentes producao (Sonnet), validacao (Sonnet) e teste (Haiku). Use quando pedirem para avançar uma etapa do cronograma do CLAUDE.md ou implementar algo novo em src/cnn/ ou src/classic/.
---

# Pipeline de uma etapa

O modelo principal é o **líder**: ele planeja, escreve a especificação, delega, decide e faz o commit. Os subagentes não fazem commit.

1. **Planejar (líder).** Ler o `CLAUDE.md` (seção 5, contrato; seção 7, cronograma) e o código que a etapa toca. Rodar `pytest` para ter a linha de base. Criar um branch da `main` no padrão `cnn/<tarefa>` ou `classic/<tarefa>`.
2. **Especificar (líder).** Escrever, para cada agente de produção, uma especificação com as assinaturas exatas. Tarefas que não tocam os mesmos arquivos rodam em paralelo; a interface entre elas fica fixada na especificação.
3. **Produção (`producao`, Sonnet).** Implementa código e testes unitários, roda o `pytest`.
4. **Validação (`validacao`, Sonnet).** Revisa contra a especificação e o contrato e roda execuções curtas com dados reais. Se reprovar, o líder decide quais achados valem e devolve ao agente de produção (de preferência o mesmo, via SendMessage, para manter o contexto). Repetir até aprovar.
5. **Execução (líder).** Treinos longos rodam em segundo plano, só com `train`/`val`.
6. **Teste (`teste`, Haiku).** Bateria final: `pytest` completo, contrato dos JSONs em `results/`, higiene do git, uso do split de teste.
7. **Fechar (líder).** Commit com a identidade do `gh`, sem coautoria do Claude; PR para a `main`.

O split `"test"` da FER-2013 só é usado uma vez por método, na avaliação final (semana 5), por decisão explícita do usuário.

Se os tipos `producao`, `validacao` e `teste` não aparecerem na lista de agentes (arquivos criados na sessão atual só carregam na próxima), use `general-purpose` com `model` igual ao do arquivo e mande o agente ler `.claude/agents/<nome>.md` antes de começar.
