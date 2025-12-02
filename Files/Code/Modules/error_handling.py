import os
import json
import pandas as pd
from typing import Dict


def registering_errors_on_generation() -> None:
    """ Register errors found during generation in a JSON file. """

    def get_errors_on_generation() -> Dict:
        JSONL_PATH = 'Generations/jsonllm.jsonl'
        os.path.exists(JSONL_PATH)

        if os.path.exists(JSONL_PATH):
            with open(JSONL_PATH, 'r') as f:
                lines = f.readlines()
                lines_to_remove = {}
                for idx, line in enumerate(lines):
                    if not line.startswith('{"id": "'):
                        lines_to_remove[idx] = json.loads(line.strip())
        return lines_to_remove

    lines_to_remove = get_errors_on_generation()
    # save dict as JSON file
    output_path = 'Generations/jsonllm_errors.json'
    json.dump(lines_to_remove, open(output_path, 'w'), indent=4)


def removing_errors_from_generation() -> None:
    """ Remove erroneous entries from the main JSONL generation file. """
    ERRORS_PATH = 'Generations/jsonllm_errors.json'
    ORIGINAL_GENERATION = 'Generations/jsonllm.jsonl'
    CLEANED_GENERATION = 'Generations/jsonllm_cleaned.jsonl'

    if os.path.exists(ERRORS_PATH):
        with open(ERRORS_PATH, 'r') as f:
            errors = json.load(f)
            error_indices = set(int(idx) for idx in errors.keys())

        with open(ORIGINAL_GENERATION, 'r') as f:
            lines = f.readlines()

        cleaned_lines = [line for idx, line in enumerate(
            lines) if idx not in error_indices]

        with open(CLEANED_GENERATION, 'w') as f:
            f.writelines(cleaned_lines)


# registering_errors_on_generation()
# removing_errors_from_generation()
