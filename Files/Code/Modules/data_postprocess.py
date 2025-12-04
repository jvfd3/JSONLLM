"""
Module for post-processing data files.
Includes functions to convert JSON files to JSONL format and save them in multiple formats.

## ToDo

- [ ] Normalize JSON keys capitalization
- [ ] Normalize JSON key ordering
- [ ] Normalize JSON key spacing/underscores
"""


import pandas as pd
from Modules.hyperparameters import get_hyperparameters

HYPER = get_hyperparameters()['load_db']


def cleaning_generated_jsonllm(df: pd.DataFrame) -> pd.DataFrame:
    original_size = df.shape[0]
    df_cleaned = df[df['values_indices'].apply(
        lambda x: isinstance(x, str)) == False]
    new_size = df_cleaned.shape[0]

    if HYPER['verbose']:
        print(f'removed {original_size - new_size} rows with generation error')

    # Resampling to keep the dataset balanced
    sampled_dfs = []

    # print(df_cleaned['dataset'].value_counts())
    min_rows = df_cleaned['dataset'].value_counts().min()
    for _, group in df_cleaned.groupby('dataset'):
        sampled_group = group.sample(n=min_rows, random_state=HYPER['seed'])
        sampled_dfs.append(sampled_group)
    equal_sampled_df = pd.concat(sampled_dfs)

    if HYPER['verbose']:
        original_size = df_cleaned.shape[0]
        new_size = equal_sampled_df.shape[0]
        print(
            f'removed {original_size - new_size} rows to rebalance the dataset')

    return equal_sampled_df


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
