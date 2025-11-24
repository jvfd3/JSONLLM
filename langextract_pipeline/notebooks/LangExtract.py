""" LangExtract Product Information Extraction Example"""

from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
import json
import langextract as lx
import textwrap

STOPPING_POINT = 1812

dataset = load_dataset("av-generation/ae-110k-dataset")
texts = dataset['train']['text']

texts = texts[STOPPING_POINT:]

output_path = "langextract_pipeline/notebooks/ae110k_extractions.jsonl"
MAX_WORKERS = 10


prompt = textwrap.dedent("""\
Extract structured product information from the text.
Identify product name, brand, model, category, color, size, material, and any key attributes.
Use the exact text for extractions — do not paraphrase.
Return relevant attributes that describe each product clearly.
""")

examples = [
    lx.data.ExampleData(
        text="Camiseta PoloTech masculina de algodão, cor azul marinho, disponível nos tamanhos M, G e GG.",
        extractions=[
            lx.data.Extraction(
                extraction_class="product",
                extraction_text="Camiseta PoloTech masculina",
                attributes={
                    "brand": "PoloTech",
                    "category": "camiseta",
                    "material": "algodão",
                    "color": "azul marinho",
                    "sizes": ["M", "G", "GG"]
                },
            ),
        ],
    ),
    lx.data.ExampleData(
        text="Tênis esportivo Nike Air Zoom branco, ideal para corrida.",
        extractions=[
            lx.data.Extraction(
                extraction_class="product",
                extraction_text="Tênis esportivo Nike Air Zoom branco",
                attributes={
                    "brand": "Nike",
                    "category": "tênis esportivo",
                    "color": "branco",
                    "intended_use": "corrida"
                },
            ),
        ],
    ),
]


def extract_text(i, text):
    try:
        result = lx.extract(
            text_or_documents=text,
            prompt_description=prompt,
            examples=examples,
            model_id="gemma2:2b",
            model_url="http://localhost:11434",
            fence_output=False,
            use_schema_constraints=False,
            language_model_params={"timeout": 900}
        )

        if not result.extractions:
            return i, None

        e = result.extractions[0]
        attrs = e.attributes or {}

        # Montar campos no mesmo formato do AE-110K
        attributes = list(attrs.keys())
        values = list(attrs.values())
        values_text = " | ".join(map(str, values))
        attributes_values = " | ".join(
            f"attribute: {k}, value: {v}" for k, v in attrs.items()
        )
        json_answer = str(attrs)  # igual ao dataset (aspas simples)
        values_indices = []

        record = {
            "id": i,
            "text": text,
            "attributes": attributes,
            "values": values,
            "values_indices": values_indices,
            "values_text": values_text,
            "attributes_values": attributes_values,
            "json_answer": json_answer,
        }
        return i, record
    except Exception as err:
        return i, {"error": str(err)}


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, open(output_path, "a") as f:
    for i, record in executor.map(lambda args: extract_text(*args), enumerate(texts, start=STOPPING_POINT)):
        if record and "error" not in record:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"[{i}] OK")
        else:
            print(f"[{i}] ERROR: {record}")
