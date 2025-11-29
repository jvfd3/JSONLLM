# Apresentação

## Slide 1: Título e Introdução

- **Título:** JSONLLM: Extração Estruturada de Atributos em E-commerce Usando LLMs
- **Problema:** A extração e organização de atributos de produtos (tamanho, cor, material) a partir de descrições textuais não estruturadas é uma tarefa desafiadora.
- **Potencial dos LLMs:** Grandes Modelos de Linguagem demonstraram potencial significativo para extração precisa e estruturada de atributos em formatos como JSON.
- **Motivação do Projeto:** A necessidade de _benchmarks_ robustos e de qualidade, pois bases existentes são incompletas, focam apenas em valores textuais e sub-representam a complexidade real das descrições de produtos.

## Slide 2: Proposta e Objetivos

- **Proposta:** Desenvolvimento do **JSONLLM**, um novo _benchmark_ de avaliação.
- **Solução:** O JSONLLM agrupa e unifica dados de três _datasets_ públicos: MAVE, AE110K e OA-Mine.
- **Inovação Principal:** Expansão do potencial avaliativo ao incluir características não abordadas pelos anteriores, como atributos com **valores não textuais** (numéricos/categóricos) e em formato de lista.
- **Objetivos:** Disponibilizar um _framework_ para metrificar o desempenho de LLMs na extração de atributos em formato JSON e comparar o novo _dataset_ com bases existentes.

## Slide 3: Metodologia - Coleta de Dados

- **Datasets Originais:** Utilização do AE-110K (Attribute Extraction 110K), OA-Mine (Open Attribute Mine) e MAVE (Multi-source Attribute Value Extraction).
- **Processamento de Dados:** Remoção de duplicatas e repartição em modelos de treino (80%), validação (10%) e teste (10%).
- **Estrutura do JSONLLM:** O _dataset_ final possui dimensões de $(90.000 \times 8)$, contendo descrições textuais únicas de produtos e seus atributos.
- **Campos Chave:** O _benchmark_ inclui campos importantes como `category`, `attributes`, `text`, e o campo alvo `json_answer` (atributos e valores em formato JSON).

## Slide 4: Metodologia - Modelos e Desafios

- **Testes Iniciais:** Uso dos modelos LiquidAI/LFM2-350M-Extract e LiquidAI/LFM2-1.2B-Extract.
- **Desafios Encontrados:** Falha na geração consistente de saídas JSON válidas e alucinações recorrentes (retornando chaves sequenciais como "atributo_1").
- **Estratégias de Mitigação:** Tentativas com _role prompting_, _few-shot learning_ e bibliotecas de _parsing_ de JSON tiveram baixa taxa de sucesso.
- **Modelo de Sucesso:** O modelo **Gemma-2-2B** da Google, auxiliado pela biblioteca **LangExtract**, obteve resultados satisfatórios na extração de JSON válido.
- **Técnicas Aplicadas (Gemma-2-2B):** _Role-prompting_, _few-shot learning_ e _Chain-of-Thought_.

## Slide 5: A Limitação dos Benchmarks e a Necessidade de Novas Métricas

- **Descoberta Central:** Durante a avaliação no AE110K, o LangExtract demonstrou desempenho superior, recuperando mais informações úteis do que as anotadas.
- **Inconsistência do Benchmark:** O AE110K revelou-se um _benchmark_ inconsistente e incompleto, com anotações superficiais, penalizando injustamente o modelo em comparações literais.
- **Necessidade de Avaliação Conceitual:** Tornou-se crucial medir a **proximidade conceitual** entre as predições do modelo e os rótulos de referência.

## Slide 6: Métricas de Avaliação Adotadas

- **Métricas Tradicionais:** Utilização de Precisão, Revocação e F1-_score_, baseadas em Verdadeiros Positivos (VC), Falsos Positivos (NV), e Falsos Negativos (VN).
- **Inovação em Avaliação:** Introdução da métrica de **Similaridade Semântica**.
- **Objetivo da Similaridade Semântica:** Permitir uma avaliação mais robusta, considerando que valores podem ser expressos de maneiras diferentes, mas possuir o mesmo significado.
- **Implementação:** Uso do modelo _encoder-only_ **MXBAI** (mixedbread-ai/mxbai-embed-large-v1) para gerar _embeddings_ e calcular a similaridade do cosseno entre valores preditos e verdadeiros.

## Slide 7: Resultados e Conclusão

- **Resultado Principal:** O _evaluation suite_ **JSONLLM** foi apresentado, suprindo limitações estruturais de bases anteriores e agregando a característica de valores em formato de lista.
- **Conclusão sobre LLMs Abertos:** Os experimentos demonstraram que modelos abertos (como Gemma-2-2B/LangExtract), combinados com técnicas adequadas de _prompting_ e avaliação semântica, alcançam resultados robustos e consistentes.
- **Impacto:** O JSONLLM possibilita avaliações mais realistas do desempenho de LLMs na extração de atributos de produtos.

| Atributo             | AE110K | MAVE | OA-Mine | JSONLLM |
| -------------------- | :----: | :--: | :-----: | :-----: |
| asin                 |        |      |    X    |    X    |
| source               |        |  X   |         |    X    |
| category             |        |  X   |    X    |    X    |
| values_indices       |   X    |  X   |    X    |         |
| attributes_values    |   X    |  X   |    X    |    X    |
| attributes           |   X    |  X   |    X    |    X    |
| json_answer          |   X    |  X   |    X    |    X    |
| text                 |   X    |  X   |    X    |    X    |
| values_text          |   X    |  X   |    X    |    X    |
| values               |   X    |  X   |    X    |    X    |
| id                   |   X    |  X   |         |    X    |
| candidate_attributes |   X    |      |         |    X    |
| candidate_example    |   X    |      |         |    X    |
| candidate_text       |   X    |      |         |    X    |

| Característica     | AE110K |   MAVE    | OA-Mine | JSONLLM |
| ------------------ | :----: | :-------: | :-----: | :-----: |
| Linhas             | 110000 | 2.073.115 |  1.938  | 90.000? |
| Colunas            |   11   |    10     |    9    |    8    |
| Valores com listas |        |           |         |   Sim   |
