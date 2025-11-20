import langextract as lx

def extract_text(i, text, prompt, examples):
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

        record = {
            "id": i,
            "text": text,
            "attributes": list(attrs.keys()),
            "values": list(attrs.values()),
            "values_indices": [],
            "values_text": " | ".join(map(str, attrs.values())),
            "attributes_values": " | ".join(
                f"attribute: {k}, value: {v}" for k, v in attrs.items()
            ),
            "json_answer": str(attrs),
        }
        return i, record

    except Exception as err:
        return i, {"error": str(err)}
