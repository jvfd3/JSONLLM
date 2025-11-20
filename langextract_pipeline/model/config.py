import textwrap
import langextract as lx

PROMPT = textwrap.dedent("""\
Extract structured product information from the text.
Identify product name, brand, model, category, color, size, material, and any key attributes.
Use the exact text for extractions — do not paraphrase.
Return relevant attributes that describe each product clearly.
""")

EXAMPLES = [
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
