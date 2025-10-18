# Coisas a fazermos

> Focar na operacionalização e nos impactos das LLMs.

- Baselines
  - [x] Descobrir quais
  - [x] Quais são as baselines clássicas?
    - R:
      - OpenTag
      - BiLSTM-CRF
      - AVEQA
      - SUOT
  - [ ] Implementar
    - R: WIP
  - [ ] Começar pelo melhor que já existe
- Dimensões de teste: O que com quem?
  1. Modelos:
     1. Modelos pré-treinados nessa tarefa (liquid.ai lfm2-extract (Upperbound), Lang Extract: framework do Google (Comercial)), Genéricos, Fine-Tuned
     2. Tamanho do modelo
     3. Famílias de LLMs
  2. Prompts: Que prompts usar? Ficar testando e se basear no que já usaram pra ver que diferença faz.
     1. Zero-Shot, Few-Shot
  3. Datasets:
     1. MAVE, TREC, AE-110K, OA-Mine
- Avaliação
  - [ ] Precisão
  - [ ] Recall
  - [ ] F1-Score
  - [ ] Procurar quais outras métricas são usadas na literatura e já estão implementadas.
  - [ ] Procurar bibliotecas.

---

- Escopo
- Fora do escopo
  - Escalabilidade para e-commerce
  - Apresentação dos produtos

---

- Dúvida
  - P: Extrair atributo de Imagem?
    - R: Não, focar em texto.
  - P: Extrair atributos de produtos similares?
    - R: Sim.
  - P: Categorizar?
    - R: Não, mas pode ser um potencial aprimoramento futuro.
  - P: Recategorizar?
    - R: NÃO, mas...
  - P: Testar em categorias diferentes?
    - R: Inicialmente só uma categoria. Potencial expansão posterior.
  - P: Quais as limitações dos datasets e das baselines?
    - R: Pesquisar posteriormente.
  - P: O que é esse autoaperfeiçoamento?
    - R: Pesquisar posteriormente.

## Tarefas

### 💠

- [ ] Baixar o dataset AE-110K e entender como ele conecta no BiLSTM-CRF.
- [ ] Aplicar os métodos de avaliação (Precisão, Recall, F1) no pipeline atual.
- [ ] Fazer o nosso (conversar depois)

### 🍯

- [ ] Clonando o BiLSTM-CRF e avaliando como conectar no dataset.
- [ ] Como conectar o BiLSTM-CRF com as métricas de avaliação.
- [ ] Fazer o nosso (conversar depois)
