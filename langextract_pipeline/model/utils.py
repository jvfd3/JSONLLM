import json
from concurrent.futures import ThreadPoolExecutor

def parallel_extract(texts, func, prompt, examples, output_path, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor, open(output_path, "w") as f:
        for i, record in executor.map(lambda args: func(*args), [(i, t, prompt, examples) for i, t in enumerate(texts)]):
            if record and "error" not in record:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"[{i}] OK")
            else:
                print(f"[{i}] ERROR: {record}")
