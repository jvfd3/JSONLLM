import json

SEP = "\u0001" # caractere presente no AE-110K

def convert_line(line: str):
    line = line.strip()
    if not line:
        return None

    if SEP in line:
        segments = line.split(SEP)
        if len(segments) < 3:
            title = segments[0]
            title_tokens = title.split()
            return title_tokens, ["O"] * len(title_tokens)

        title = segments[0].strip()
        title_tokens = title.split()
        tokens = list(title_tokens)
        labels = ["O"] * len(title_tokens)

        rest = segments[1:]
        n_pairs = (len(rest) // 2) * 2
        for i in range(0, n_pairs, 2):
            attr = rest[i].strip()
            value = rest[i+1].strip()
            if not attr or not value or value == "NULL":
                continue
            val_tokens = value.split()
            for j, tok in enumerate(val_tokens):
                labels.append(f"B-{attr}" if j == 0 else f"I-{attr}")
                tokens.append(tok)
        return tokens, labels

    parts = line.split()
    tokens, labels = [], []
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            value = value.strip()
            if not value or value == "NULL":
                continue
            for j, tok in enumerate(value.split()):
                tokens.append(tok)
                labels.append(f"B-{key}" if j == 0 else f"I-{key}")
        else:
            tokens.append(part)
            labels.append("O")
    return tokens, labels

def convert_file(input_path: str, output_path: str):
    total, kept = 0, 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            total += 1
            res = convert_line(line)
            if not res:
                continue
            tokens, labels = res
            if len(tokens) != len(labels) or len(tokens) == 0:
                continue
            kept += 1
            f_out.write(
                json.dumps(tokens, ensure_ascii=False) + "\t" +
                json.dumps(labels, ensure_ascii=False) + "\n"
            )

convert_file("AE-110K.txt", "corpus_dir/datasetae110k.txt")