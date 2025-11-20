import json

def convert_to_ae110k_format(input_path, output_path):
    converted = []
    with open(input_path) as f:
        for line in f:
            obj = json.loads(line)
            attr_str = obj.get("attributes_values", "")
            parts = [p.strip() for p in attr_str.split("|") if "attribute:" in p]
            pairs = {}
            for p in parts:
                if "attribute:" in p and "value:" in p:
                    a = p.split("attribute:")[1].split(",")[0].strip()
                    v = p.split("value:")[1].strip()
                    if a and v and v.lower() != "none":
                        pairs[a] = v

            json_answer = "{" + ", ".join([f"'{k}': '{v}'" for k, v in pairs.items()]) + "}"
            converted.append({"json_answer": json_answer})

    with open(output_path, "w") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
