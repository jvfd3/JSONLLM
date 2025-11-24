# Dataset OAMine - subconjunto Amazon

Inicialmente avaliamos o modelo de extração de atributos com o benchmark AE110k. No entanto, observamos problemas de incompletude e inconsistência estrutural nas anotações, que penalizavam injustamente o desempenho do modelo, especialmente em categorias como fabric/material, brand e category.

Como alternativa, adotamos o dataset OAMine (subset Amazon), que contém descrições reais de produtos da Amazon, organizadas por categoria (por exemplo, toothbrush, breakfast_cereal, coffee, sunscreen etc.). Para algumas dessas categorias, o dataset fornece anotações manuais em nível de entidade (span-based). Cada entrada contém o identificador do produto (asin), o título e uma lista de entidades anotadas, com rótulos como Brand, Flavor, Net content, Color, entre outros.

A partir desses arquivos, construímos um benchmark próprio no formato JSONLLM, unindo:
(i) os textos completos dos produtos (título, bullet points e descrição) a partir de raw/*.jsonl; e
(ii) os atributos anotados em annotations/*.jsonl.

O resultado é um arquivo oamine_benchmark_jsonllm.jsonl, em que cada linha representa um produto:

Nem todas as categorias presentes em raw/ possuem anotações em annotations/. Por isso, o benchmark final é formado apenas pelas categorias com ground truth disponível (por exemplo, toothbrush, safety_mask, breakfast_cereal, laundry_detergent, shoes, conditioner, sunscreen, coffee, tea e vitamin). Isso resulta em aproximadamente 2.000 exemplos com anotações completas, o que já representa um cenário bem mais realista e consistente do que o AE110k

Extração de atributos com LangExtract

Utilizamos o LangExtract como modelo principal de extração, configurado com few-shot e raciocínio em cadeia (chain-of-thought) sobre as descrições de produtos. Para cada entrada do benchmark, passamos o campo text para o LangExtract, com um prompt descritivo e um conjunto fixo de exemplos anotados. O modelo de linguagem subjacente é um LLM open-source (Gemma 2B, servido via Ollama), mantendo a restrição de uso apenas de modelos abertos.

O LangExtract retorna, para cada produto, um objeto JSON com atributos inferidos (por exemplo, brand, category, flavor, net_content, color, material). Esse JSON é serializado no campo json_answer de um arquivo de predições (oamine_predictions.jsonl), que é então comparado ao ground_truth do benchmark.

Avaliação semântica com MXBAI

Como as chaves de atributos (labels) podem variar entre o ground truth e o JSON gerado pelo modelo (por exemplo, Flavor vs flavor), e os valores podem ter variações lexicais (“raisin granola” vs “raisin granola cereal”), utilizamos uma avaliação baseada em similaridade semântica, em vez de string match literal.

Usamos o modelo de embeddings mixedbread-ai/mxbai-embed-large-v1 para codificar os valores dos atributos, e avaliamos o modelo com três métricas:

Recovery Score (RS): proporção de atributos do ground_truth que o modelo recupera corretamente (similaridade cosseno >= 0,55).

Enrichment Score (ES): proporção de atributos adicionais gerados pelo modelo que não estão no ground_truth, mas representam informação potencialmente relevante.

Ground Truth Completeness Gap (GTCG): quão incompleto o ground truth é em relação às predições do modelo, definido como ES / (RS + ES).

Essas métricas permitem comparar diferentes variantes do LangExtract (few-shot, CoT, constrained decoding, memory bank etc.) de forma justa, mesmo em cenários onde o catálogo de atributos é aberto e o ground truth não cobre todos os aspectos presentes no texto.
