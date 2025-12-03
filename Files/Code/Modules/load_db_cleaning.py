"""
Functions:
1. get_specific_cols_to_remove: Returns the specific columns to remove for each dataset
2. keep_larger_json_answer: Keeps only the rows with the larger json_answer for each text
3. general_cleaning: General cleaning steps
4. clean_dataset: Cleans the dataset by removing duplicates and NaN values
5. split_df: Splits the dataframe into train, validation and test sets based on given proportions
6. clean_dataframes: Cleans the dataframes according to the dataset
7. join_datasets: Joins all datasets into a single DataFrame and adds a 'dataset' column
8. clean_candidates: Cleans the candidates in the joined datasets by removing rows with null candidates
9. debug_check_duplicated_ids: Check for duplicated 'id' values in the DataFrame

## Sequence of operations:

- clean_dataframes
    - clean_dataset
        - get_specific_cols_to_remove
        - clean_candidates
        - general_cleaning
    - split_df
---

- join_datasets
    - get_all_dataset_columns
    - keep_larger_json_answer

---



"""


import pandas as pd
from typing import Dict
from Modules.hyperparameters import get_hyperparameters

HYPER = get_hyperparameters()['load_db']


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
    print(f"Getting specific columns to remove for dataset: {dataset_name}")
    if HYPER['simple_columns'] and dataset_name in COLS_TO_REMOVE:
        print(f"Removing specific columns for dataset: {dataset_name}")
        return COLS_TO_REMOVE[dataset_name]
    return []


def general_cleaning(cleaning_dataset: pd.DataFrame) -> pd.DataFrame:
    """ General cleaning steps """
    if HYPER['verbose']:
        print("General Clean: drop_na, keep bigger json_answer, re-index")

    # Dropping NaN values
    # cleaning_dataset = cleaning_dataset.dropna()

    # Dropping exact matches
    # cleaning_dataset = cleaning_dataset.drop_duplicates()

    # Resorting by text
    # cleaning_dataset = cleaning_dataset.sort_values('text')

    # Re-indexing
    cleaning_dataset = cleaning_dataset.reset_index(drop=True)

    return cleaning_dataset


def clean_dataset(cleaning_dataset: pd.DataFrame, dataset_name: str = 'mave') -> pd.DataFrame:
    """ Cleans the dataset by removing duplicates and NaN values """

    original_shape = cleaning_dataset.shape

    if HYPER['simple_columns']:
        # Removing specific columns
        cols_to_remove = get_specific_cols_to_remove(dataset_name)
        cleaning_dataset = cleaning_dataset.drop(columns=cols_to_remove)

    if dataset_name in ['ae-110k']:
        cleaning_dataset = clean_candidates(cleaning_dataset)

    # General cleaning
    cleaning_dataset = general_cleaning(cleaning_dataset)

    # Defining the dataset column
    cleaning_dataset['dataset'] = dataset_name

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

    test_df = shuffled_df.iloc[: size_test].copy()
    validation_df = shuffled_df.iloc[size_test: size_test +
                                     size_validation].copy()
    train_df = shuffled_df.iloc[size_test + size_validation:].copy()

    dfs = {'test': test_df, 'validation': validation_df, 'train': train_df}

    for split, df in dfs.items():
        dfs[split]['split'] = split

    return dfs


def clean_dataframes(dfs: Dict[str, pd.DataFrame], dataset_name: str = 'ae-110k') -> Dict[str, pd.DataFrame]:
    """ Cleans the dataframes according to the dataset """
    # Joining all splits into one dataframe for cleaning
    joined_df = pd.concat(dfs.values(), ignore_index=True)
    cleaned_dataset = clean_dataset(joined_df, dataset_name)
    dfs = split_df(cleaned_dataset, HYPER['proportions'])
    return dfs


def clean_candidates(cleaning_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the candidates in the joined datasets by removing rows with null candidates
    and replacing specific placeholder values with NaN.

    Args:
        joined_datasets (pd.DataFrame): The joined datasets DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with valid candidates.

    ToDo:
    - [ ] The candidate cleaning should be run only on datasets that have candidates, before joining.
    """

    def clean_exact(d: dict | list, target: str = "{'json_answer': '', 'text': ''}") -> str | None:
        """ Cleans exact matches of a target string in a dictionary or list. """
        return pd.NA if str(d) == target else d

    # Replace candidate_attributes `['']` with Missing Value;
    if 'candidate_attributes' in cleaning_df.columns:
        cleaning_df['candidate_attributes'] = cleaning_df['candidate_attributes'].apply(
            lambda x: clean_exact(x, "['']"))

    # Replace candidate_example `{'json_answer': '', 'text': ''}` with Missing Value;
    if 'candidate_example' in cleaning_df.columns:
        cleaning_df['candidate_example'] = cleaning_df['candidate_example'].apply(
            clean_exact)

    # Replace candidate_text '' with Missing Value
    if 'candidate_text' in cleaning_df.columns:
        cleaning_df[['candidate_text']] = cleaning_df[[
            'candidate_text']].replace('', pd.NA)

    # Show only rows where not all candidate columns are null
    # debug_candidate = cleaning_df[
    #     cleaning_df[['candidate_attributes',
    #                  'candidate_example', 'candidate_text']].notnull().any(axis=1)
    # ]
    # return debug_candidate

    return cleaning_df


def keep_larger_json_answer(df: pd.DataFrame) -> pd.DataFrame:
    """ Keeps only the rows with the larger json_answer for each text """
    if HYPER['verbose']:
        print("Keeping only rows with larger json_answer for each text")

    df['json_len'] = df['json_answer'].str.len()
    df = (
        df
        .sort_values('json_len', ascending=False)
        .drop_duplicates(subset='text', keep='first')
        .drop(columns=['json_len'])
    )
    return df


def join_datasets(datasets: dict) -> pd.DataFrame:
    """
    Joins all datasets into a single DataFrame and adds a 'dataset' column.

    Args:
        datasets (dict): Dictionary containing datasets.

    Returns:
        pd.DataFrame: Combined DataFrame with an additional 'dataset' column.

    ---

    ToDo:
    - [ ] Turn categorical columns into correct data type
    - [ ] Process duplicate key-values into list of values
    """

    def get_all_dataset_columns(datasets_dict: Dict[str, pd.DataFrame]) -> set:
        """
            Get a sorted list of all unique column names across all datasets.
        """

        cols = set()
        for dataset in datasets_dict.values():
            columns = list(dataset['train'].columns)
            cols.update(columns)
        if HYPER['verbose']:
            print("All unique columns across datasets:")
            print(sorted(cols))
        return list(cols)

    all_columns = get_all_dataset_columns(datasets)

    joined_dfs = pd.DataFrame(columns=list(all_columns))

    for _, dataset_splits in datasets.items():
        for _, df in dataset_splits.items():
            # print(f"df['{dataset_name}]['{split_name}'].shape: {df.shape}")
            df_copy = df.copy()
            # df_copy['dataset'] = dataset_name
            # df_copy['split'] = split_name
            joined_dfs = pd.concat([joined_dfs, df_copy], ignore_index=True)

    # Only keep the larger json_answer if there are duplicates
    joined_dfs = keep_larger_json_answer(joined_dfs)

    # Sort the final DataFrame by 'text' column for consistency
    # joined_dfs = joined_dfs.sort_values(by='text').reset_index(drop=True)

    # if 'asin' has value replace 'id' value with it
    if 'id' in joined_dfs.columns and 'asin' in joined_dfs.columns:
        joined_dfs['id'] = joined_dfs['asin'].fillna(joined_dfs['id'])

        # drop 'asin' column
        joined_dfs = joined_dfs.drop(columns=['asin'])

    # Re-ordering columns
    columns_order = [
        'id',
        'dataset',
        'split',
        'text',
        'json_answer',
        'source',
        'category',
        'attributes',
        'attributes_values',
        'values',
        'values_indices',
        'values_text'
        'candidate_attributes',
        'candidate_example',
        'candidate_text',
        # 'asin', # asin became id
    ]

    # get the intersection of columns_order and joined_dfs.columns
    safe_columns_order = [
        col for col in columns_order if col in joined_dfs.columns]

    joined_dfs = joined_dfs.reindex(safe_columns_order, axis=1)

    # Sort by text and reindex
    joined_dfs = joined_dfs.sort_values('text').reset_index(drop=True)

    return joined_dfs


def debug_check_duplicated_ids(df: pd.DataFrame) -> None:  # Not used
    """
    Check for duplicated 'id' values in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to check.

    Returns:
        pd.DataFrame: DataFrame containing duplicated 'id' rows.
    """
    duplicated_ids = df[df.duplicated(subset=['id'], keep=False)]
    # .sort_values(by='id')
    print(
        f"Number of duplicated 'id' entries: {duplicated_ids['id'].nunique()}")
    # print(duplicated_ids[['id', 'dataset', 'text']])
    # I don't know why there are so many duplicated ids
