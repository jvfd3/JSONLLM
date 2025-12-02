"""
Module for post-processing data files.
Includes functions to convert JSON files to JSONL format and save them in multiple formats.
"""


import pandas as pd
# import os


def save_jsonl(input_path: str = 'Generations/jsonllm_cleaned.jsonl', output_path: str = 'Datasets/jsonllm/jsonllm'):
    """ Convert a JSON file to JSONL format. """

    df = pd.read_json(input_path, lines=True)
    df.to_csv(output_path+'.csv', index=False)
    df['values'] = df['values'].astype(str)
    df.to_parquet(output_path+'.parquet', index=False)
    print(f"Saved JSONL to {output_path}.csv, and {output_path}.parquet")

# input_path = 'Generations/jsonllm_cleaned.jsonl'
# output_path = 'Datasets/jsonllm/jsonllm'

# save_jsonl(input_path, output_path)
