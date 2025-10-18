""" Loading DB """

import pandas as pd
from datasets import load_dataset

def get_datasets(selected_df: str) -> pd.DataFrame:
    """ selected_df options: 'ae-110k', 'oa-mine', 'mave' """

    def get_file_name(segment: str) -> str:
        """ Returns the file name for the given segment ('train' or 'test') """
        file_name = f'data/{segment}-00000-of-00001.parquet'
        return file_name

    def get_url(dataset: str, segment: str) -> str:
        """ Constructs the URL for the given dataset and segment ('train' or 'test') """
        base_url = f'hf://datasets/av-generation/{dataset}-dataset/'
        file_name = get_file_name(segment)
        url = base_url + file_name
        return url

    def load_parquet(dataset: str, segment: str) -> pd.DataFrame:
        """ Loads the parquet file from the given dataset and segment ('train' or 'test') """
        url = get_url(dataset, segment)
        loaded_df = pd.read_parquet(url)
        return loaded_df

    def load_dataset_from_hf(dataset: str) -> pd.DataFrame:
        """ Loads the dataset from Hugging Face """
        ds = load_dataset(f'av-generation/{dataset}-dataset')
        return ds

    dfs = {
        'train': pd.DataFrame(),
        'test': pd.DataFrame()
    }

    dataset_options = ['ae-110k', 'oa-mine', 'mave'] # Check if mave works

    if selected_df in dataset_options:
        dfs = load_dataset_from_hf(selected_df)
        # dfs['train'] = load_parquet(selected_df, 'train')
        # dfs['test'] = load_parquet(selected_df, 'test')

    return dfs

# load_database()
