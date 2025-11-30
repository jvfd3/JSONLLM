""" Loading DB

- To-Do's
  - [X] Save dataframes locally after first load
  - [X] Add option to load dataframes from local path
  - [X] Join all train, validation and test sets into single dataframes, clean and them split them again with same proportions
"""

""" Libraries Imports """

# from Modules.hyperparameters import get_hyperparameters  # For module testing
import os
import pandas as pd
import datasets as ds
from .load_db_cleaning import clean_dataframes
from typing import Dict
from .hyperparameters import get_hyperparameters  # For module running

# from hyperparameters import get_hyperparameters  # When local running
# from ds import load_dataset

# """ Constants """

HYPER = get_hyperparameters()['load_db']
SPLITS = HYPER['proportions'].keys()  # ['train', 'test', 'validation']


def get_dataframes(selected_df: str) -> Dict[str, pd.DataFrame]:
    """ selected_df options: 'ae-110k', 'oa-mine', 'mave' """

    # def load_parquet(dataset: str, segment: str) -> pd.DataFrame:
    #     """ Loads the parquet file from the given dataset and segment ('train' or 'test') """
    #     def get_file_name(segment: str) -> str:
    #         """ Returns the file name for the given segment ('train' or 'test') """
    #         file_name = f'data/{segment}-00000-of-00001.parquet'
    #         return file_name
    #     def get_url(dataset: str, segment: str) -> str:
    #         """ Constructs the URL for the given dataset and segment ('train' or 'test') """
    #         base_url = f'hf://datasets/av-generation/{dataset}-dataset/'
    #         file_name = get_file_name(segment)
    #         url = base_url + file_name
    #         return url
    #     url = get_url(dataset, segment)
    #     loaded_df = pd.read_parquet(url)
    #     return loaded_df

    # def load_datasets_from_path(saved_df_path: str) -> ds.dataset_dict.DatasetDict:
    #     """ Loads the dataset from local path """
    #     if HYPER['verbose']:
    #         print(f"Loading dataset from local path: {df_saving_path}")
    #     dfs = ds.load_from_disk(saved_df_path)
    #     return dfs

    # def save_dataset_to_path(datasets: ds.dataset_dict.DatasetDict, saving_path: str) -> None:
    #     """ Saves the dataset to local path """
    #     datasets.save_to_disk(saving_path, max_shard_size='100MB')

    def load_dataframes_from_path(saved_df_path: str) -> Dict[str, pd.DataFrame]:
        """ Loads the dataframes from local path """
        if HYPER['verbose']:
            print(f"Loading dataframes from local path: {df_saving_path}")
        dataframes = {split: pd.DataFrame() for split in SPLITS}

        for split in SPLITS:
            file_path = os.path.join(saved_df_path, f"{split}.parquet")
            if os.path.exists(file_path):
                dataframes[split] = pd.read_parquet(file_path)
        return dataframes

    def load_dataframes_from_hf(selected_df: str) -> Dict[str, pd.DataFrame]:
        """ Loads the dataset from Hugging Face """
        def convert_to_pandas(dfs: ds.dataset_dict.DatasetDict) -> Dict[str, pd.DataFrame]:
            """ Converts the dataset splits to pandas DataFrames """
            pandas_dfs = {}

            for split in SPLITS:
                if split in dfs.keys():
                    pandas_dfs[split] = dfs[split].to_pandas()

            return pandas_dfs
        if HYPER['verbose']:
            print(
                f"Loading dataframes from Hugging Face for dataset: {selected_df}")
        datasets = ds.load_dataset(f'av-generation/{selected_df}-dataset')
        pandas_dfs = convert_to_pandas(datasets)
        return pandas_dfs

    def get_path_to_save(selected_df: str = '') -> str:
        """ Returns the path of the running code file """
        # current_path = os.path.abspath(__file__)
        saving_path = 'JSONLLM/Files/Code/Datasets/'
        normalized_path = os.getcwd().split('JSONLLM')[0] + saving_path
        normalized_path = os.path.normpath(normalized_path)
        normalized_path = os.path.join(normalized_path, selected_df)
        return normalized_path

    def save_dataframe_to_path(dfs: Dict[str, pd.DataFrame], saving_path: str) -> None:
        """ Saves the dataframe to local path """
        if HYPER['verbose']:
            print(f"Saving dataframes to local path: {saving_path}")
        os.makedirs(saving_path, exist_ok=True)
        for split, df in dfs.items():
            if df is None or df.empty:
                continue
            out = os.path.join(saving_path, f"{split}.parquet")
            df.reset_index(drop=True).to_parquet(out, index=False)

    dfs = {split: pd.DataFrame() for split in SPLITS}

    dataset_options = ['ae-110k', 'oa-mine', 'mave']

    df_saving_path = get_path_to_save(selected_df)
    if selected_df in dataset_options:
        if HYPER['verbose']:
            print(f"Loading dataframe: {selected_df}")
        if os.path.exists(df_saving_path):
            dfs = load_dataframes_from_path(df_saving_path)
            # Re-cleaning just in case
            dfs = clean_dataframes(dfs, selected_df)
        else:
            dfs = load_dataframes_from_hf(selected_df)
            dfs = clean_dataframes(dfs, selected_df)
            save_dataframe_to_path(dfs, df_saving_path)

    if HYPER['verbose']:
        print(f'Loaded dataframes with splits: {list(dfs.keys())}')
        shapes = {split: df.shape for split, df in dfs.items()}
        print(f'The shapes of the dataframes are: {shapes}')
    return dfs


def get_dataframe_dict(selected_dataframes: list[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
    """ Returns a dictionary of dataframes for the selected datasets """
    dataframes_dict = {}

    for dataset_name in selected_dataframes:
        dataframes_dict[dataset_name] = get_dataframes(dataset_name)

    return dataframes_dict

# """ Datasets """


# datasets_dict = get_dataframe_dict([
    # 'ae-110k',
    # 'oa-mine',
    # 'mave',
# ])
