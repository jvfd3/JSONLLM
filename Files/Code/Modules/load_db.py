""" Loading DB

- To-Do's
  - [X] Save dataframes locally after first load
  - [X] Add option to load dataframes from local path
  - [X] Join all train, validation and test sets into single dataframes, clean and them split them again with same proportions
"""

""" Libraries Imports """

# from hyperparameters import get_hyperparameters  # When local running
# from Modules.hyperparameters import get_hyperparameters  # For module testing
from typing import Dict
from .hyperparameters import get_hyperparameters  # For module running
import datasets as ds
import pandas as pd
import os
# from ds import load_dataset

# """ Constants """

HYPER = get_hyperparameters()['load_db']
SPLITS = HYPER['proportions'].keys()  # ['train', 'test', 'validation']

# """ CLEANING FUNCTIONS """


def get_specific_cols_to_remove(dataset_name: str) -> list:
    """ Returns the specific columns to remove for each dataset """
    COLS_TO_REMOVE = {
        'mave': [
            'id',  # Maybe needed for some reference later
            'category',  # Maybe needed for distinguishing later
            # 'text', # MUST use
            'source',  # Not dealing with char range issues
            'attributes',  # Not splitting by attributes
            'values',  # Not splitting by values
            'values_indices',  # Not dealing with char range issues
            'values_text',  # No need to split the values with a "|"
            'attributes_values',  # Not JSON enough
            # 'json_answer' # MUST use as Ground Truth
        ],
        'ae-110k': [
            'id',
            # 'text',  # MUST use as product description
            'attributes',  # No need to keep only the attributes
            'values',  # No need to keep only the values
            'values_indices',  # Not dealing with char range issues
            'values_text',  # No need to split the values with a "|"
            'attributes_values',  # Not JSON enough
            # 'json_answer', # MUST use as Ground Truth
            'candidate_attributes',  # I don't even know what theses candidates are
            'candidate_text',  # I don't even know what theses candidates are
            'candidate_example'  # I don't even know what theses candidates are
        ],
        'oa-mine': [
            'asin',  # What is ASIN? Answer: Amazon Standard Identification Number
            # 'text', # MUST use as product description
            'category',  # Maybe needed for distinguishing later
            'attributes',  # No need to keep only the attributes
            'values',  # No need to keep only the values
            'values_indices',  # Not dealing with char range issues
            'values_text',  # No need to split the values with a "|"
            'attributes_values',  # Not JSON enough
            # 'json_answer' # MUST use as Ground Truth
        ]
    }
    if dataset_name in COLS_TO_REMOVE:
        return COLS_TO_REMOVE[dataset_name]
    return []


def general_cleaning(cleaning_dataset: pd.DataFrame) -> pd.DataFrame:
    """ General cleaning steps """
    if HYPER['verbose']:
        print("General Clean: drop_na, keep bigger json_answer, re-index")

    # Dropping NaN values
    cleaning_dataset = cleaning_dataset.dropna()
    # Dropping exact matches
    # cleaning_dataset = cleaning_dataset.drop_duplicates()
    # Dropping rows with bigger JSON Answers
    cleaning_dataset['json_len'] = cleaning_dataset['json_answer'].str.len(
    )
    cleaning_dataset = (
        cleaning_dataset
        .sort_values('json_len', ascending=False)
        .drop_duplicates(subset='text', keep='first')
        .drop(columns=['json_len']
              )
    )
    # Resorting by text
    cleaning_dataset = cleaning_dataset.sort_values('text')
    # Re-indexing
    cleaning_dataset = cleaning_dataset.reset_index(drop=True)
    return cleaning_dataset


def clean_dataset(cleaning_dataset: pd.DataFrame, dataset_name: str = 'mave') -> pd.DataFrame:
    """ Cleans the dataset by removing duplicates and NaN values """

    original_shape = cleaning_dataset.shape

    # Removing specific columns
    cols_to_remove = get_specific_cols_to_remove(dataset_name)
    cleaning_dataset = cleaning_dataset.drop(columns=cols_to_remove)

    # General cleaning
    cleaning_dataset = general_cleaning(cleaning_dataset)

    if HYPER['verbose']:
        print(
            f"Shape Change: from {original_shape} to {cleaning_dataset.shape}")
    return cleaning_dataset


def split_df(dataset: pd.DataFrame, proportions: Dict[str, float]) -> Dict[str, pd.DataFrame]:
    """ Splits the dataframe into train, validation and test sets based on given proportions """
    if HYPER['verbose']:
        print(
            f"Splitting dataframe into train, validation and test sets with proportions: {proportions}")

    # Validating proportions
    total = sum(proportions.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Proportions must sum to 1.0. Got: {total}")

    shuffled_df = dataset.sample(
        frac=1, random_state=HYPER['seed']).reset_index(drop=True)

    size = len(shuffled_df)

    size_validation = int(size * proportions['validation'])
    size_test = int(size * proportions['test'])
    size_train = size - size_validation - size_test

    dfs = {
        'test': shuffled_df.iloc[: size_test],
        'validation': shuffled_df.iloc[size_test: size_test + size_validation],
        'train': shuffled_df.iloc[size_test + size_validation:],
    }
    return dfs


def clean_dataframes(dfs: Dict[str, pd.DataFrame], dataset_name: str = 'ae-110k') -> Dict[str, pd.DataFrame]:
    """ Cleans the dataframes according to the dataset """
    # Joining all splits into one dataframe for cleaning
    joined_df = pd.concat(dfs.values(), ignore_index=True)
    cleaned_dataset = clean_dataset(joined_df, dataset_name)
    dfs = split_df(cleaned_dataset, HYPER['proportions'])
    return dfs


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
        else:
            dfs = load_dataframes_from_hf(selected_df)
            dfs = clean_dataframes(dfs, selected_df)
            save_dataframe_to_path(dfs, df_saving_path)

    if HYPER['verbose']:
        print(f'Loaded dataframes with splits: {list(dfs.keys())}')
        shapes = {split: df.shape for split, df in dfs.items()}
        print(f'The shapes of the dataframes are: {shapes}')
    return dfs


# """ Datasets """

# datasets_dict = {
#     'ae-110k': get_dataframes('ae-110k'),
#     'oa-mine': get_dataframes('oa-mine'),
#     'mave': get_dataframes('mave'),
# }
