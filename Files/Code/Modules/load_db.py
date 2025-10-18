""" Loading DB """

import os
import pandas as pd
import datasets as ds
# from ds import load_dataset

def get_dataset(selected_df: str) -> ds.dataset_dict.DatasetDict:
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

    def load_from_path(saved_df_path: str) -> ds.dataset_dict.DatasetDict:
        """ Loads the dataset from local path """
        dfs = ds.load_from_disk(saved_df_path)
        return dfs

    def load_dataset_from_hf(dataset: str) -> ds.dataset_dict.DatasetDict:
        """ Loads the dataset from Hugging Face """
        dataset = ds.load_dataset(f'av-generation/{dataset}-dataset')
        return dataset

    def get_path_to_save(selected_df: str = '') -> str:
        """ Returns the path of the running code file """
        # current_path = os.path.abspath(__file__)
        saving_path = 'JSONLLM/Files/Code/Datasets/'
        normalized_path = os.getcwd().split('JSONLLM')[0] + saving_path
        normalized_path = os.path.normpath(normalized_path)
        normalized_path = os.path.join(normalized_path, selected_df)
        return normalized_path

    def save_to_path(dfs: ds.dataset_dict.DatasetDict, saving_path: str) -> None:
        """ Saves the dataset to local path """
        # print(f'Saving dataset to "{normalized_path}"...')
        dfs.save_to_disk(saving_path, max_shard_size='100MB')

    dfs = {
        'train': pd.DataFrame(),
        'test': pd.DataFrame(),
        'validation': pd.DataFrame(),
    }

    dataset_options = ['ae-110k', 'oa-mine', 'mave'] # Check if mave works
    df_saving_path = get_path_to_save(selected_df)
    if selected_df in dataset_options:
        if os.path.exists(df_saving_path):
            dfs = load_from_path(df_saving_path)
        else:
            dfs = load_dataset_from_hf(selected_df)
            save_to_path(dfs, df_saving_path)
        # dfs['train'] = load_parquet(selected_df, 'train')
        # dfs['test'] = load_parquet(selected_df, 'test')

    return dfs

# datasets_dict = {
#     'ae-110k': get_dataset('ae-110k'),
#     'oa-mine': get_dataset('oa-mine'),
#     'mave': get_dataset('mave'),
# }
