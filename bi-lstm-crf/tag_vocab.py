import json

dataset_path = "corpus_dir/datasetae110k.txt"
vocab_out = "corpus_dir/vocab.json"
tags_out = "corpus_dir/tags.json"

vocab_set = set()
tags_set = set()

with open(dataset_path, "r") as f:
    for line in f:
        if not line.strip():
            continue
        tokens_json, tags_json = line.strip().split("\t")
        tokens = json.loads(tokens_json)
        tags = json.loads(tags_json)
        vocab_set.update(tokens)
        tags_set.update(tags)

vocab = sorted(list(vocab_set))
tags = sorted(list(tags_set))

with open(vocab_out, "w") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)
with open(tags_out, "w") as f:
    json.dump(tags, f, ensure_ascii=False, indent=2)
