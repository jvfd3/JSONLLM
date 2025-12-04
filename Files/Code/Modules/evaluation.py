""" Module for evaluating model solutions and plotting results. """


from .hyperparameters import get_hyperparameters
from .Models.mixedbread import batch_embed
from typing import Dict
import ast
import json
import numpy as np
import os
import pandas as pd

HYPER = get_hyperparameters()['evaluation']


def evaluate_solution(model_solutions: Dict, reference_solutions: Dict) -> Dict:
    """ Evaluates the model
    Evaluation based on: https://arxiv.org/pdf/2310.12537

    1. NN: No predicted value, No groud truth value
    2. NV: Predicted value, No ground truth value
    3. VN: No predicted value, ground truth value
    4. VC: predicted value matches ground truth value
    5. VW: predicted value does not match ground truth value

    ---
    - VC: (TP) True Positive
    - NV: (FP) False Positive
    - VN: (FN) False Negative
    - NN: (TN) True Negative (not used in precision/recall/F1 calculations)
    ---
    Usually:
    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    ---
    Precision:  P  = VC / (NV + VC + VW)
    Recall:     R  = VC / (VN + VC + VW)
    F1-score:   F1 = 2PR / (P + R)

    """
    def count_cases(model_solutions: Dict, reference_solutions: Dict) -> Dict:
        """ Counts the different cases between model and reference solutions """
        counts = {'NN': 0, 'NV': 0, 'VN': 0, 'VC': 0, 'VW': 0}
        model_keys = set(model_solutions.keys())
        reference_keys = set(reference_solutions.keys())
        all_keys = model_keys.union(reference_keys)

        for key in all_keys:
            predicted_value = model_solutions.get(key)
            reference_value = reference_solutions.get(key)

            if predicted_value is None and reference_value is None:
                counts['NN'] += 1
            elif predicted_value is not None and reference_value is None:
                counts['NV'] += 1
            elif predicted_value is None and reference_value is not None:
                counts['VN'] += 1
            elif predicted_value == reference_value:
                counts['VC'] += 1
            else:
                counts['VW'] += 1

        return counts

    def get_precision(counts: Dict) -> float:
        """ Calculates precision """
        nv = counts['NV']
        vc = counts['VC']
        vw = counts['VW']
        all_predicted = nv + vc + vw
        if all_predicted == 0:
            return 0.0
        return vc / all_predicted

    def get_recall(counts: Dict) -> float:
        """ Calculates recall """
        vn = counts['VN']
        vc = counts['VC']
        vw = counts['VW']
        all_ground_truth = vn + vc + vw
        if all_ground_truth == 0:
            return 0.0
        return vc / all_ground_truth

    def get_f1(precision: float, recall: float) -> float:
        """ Calculates F1-score """
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    counts = count_cases(model_solutions, reference_solutions)

    precision = get_precision(counts)
    recall = get_recall(counts)
    scores = {
        'VN': int(counts['VN']),
        'NV': int(counts['NV']),
        'VC': int(counts['VC']),
        'VW': int(counts['VW']),
        'NN': int(counts['NN']),
        # 'accuracy': 0.0,
        'precision': precision,
        'recall': recall,
        'f1': get_f1(precision, recall),
    }

    return scores


def evaluate_solution_df(df: pd.DataFrame) -> pd.DataFrame:
    """ Evaluates the model solutions in a dataframe. """

    df = df.copy()

    # Temporarily convert JSON strings to dicts
    df["gt_dict"] = df["gt_json_answer"].apply(json.loads)
    df["lx_dict"] = df["lx_json_answer"].apply(json.loads)

    # Evaluates each row
    results = df.apply(
        lambda row: evaluate_solution(row["gt_dict"], row["lx_dict"]),
        axis=1
    )

    # Drop temporary dict columns
    df = df.drop(columns=["gt_dict", "lx_dict"])

    # Converts results to DataFrame
    metrics_df = pd.DataFrame(results.tolist(), index=df.index)

    # Ensure integer columns are of type int
    int_columns = ['VN', 'NV', 'VC', 'VW', 'NN']
    metrics_df[int_columns] = metrics_df[int_columns].astype(int)

    # Concatenate original df with metrics_df
    df = df.join(metrics_df)
    # df = pd.concat([df, metrics_df], axis=1)

    return df


def get_distribution_metrics(sims: np.ndarray, title: str = '') -> dict[str, float]:
    """ Print similarity metrics. """

    metrics = {
        'mean_similarity': np.mean(sims),
        'median_similarity': np.median(sims),
        'std_similarity': np.std(sims),
        'min_similarity': np.min(sims),
        'max_similarity': np.max(sims),
    }

    if HYPER['debug']:
        if title:
            print(f"\n{title}")
        print(f"Mean:\t{metrics['mean_similarity']:.4f}")
        print(f"Median:\t{metrics['median_similarity']:.4f}")
        print(f"Std:\t{metrics['std_similarity']:.4f}")
        print(f"Min:\t{metrics['min_similarity']:.4f}")
        print(f"Max:\t{metrics['max_similarity']:.4f}")

    return metrics


def get_semantic_similarity_eval(df: pd.DataFrame) -> pd.DataFrame:
    """ Full evaluation computing embeddings and similarities. """

    def check_if_embeddings_exist() -> bool:
        files = [
            'Embeddings/embeddings_text.npy',
            'Embeddings/embeddings_gt_json.npy',
            'Embeddings/embeddings_lx_json.npy',
            'Similarities/text_vs_lx.npy',
            'Similarities/text_vs_gt.npy',
            'Similarities/gt_vs_lx.npy',
        ]
        return all(os.path.exists(f) for f in files)

    if not HYPER['force_recompute_embeddings'] and check_if_embeddings_exist():
        if HYPER['debug']:
            print("Embeddings and similarities already exist. Loading from disk...")
        embeddings = {
            'text': np.load('Embeddings/embeddings_text.npy'),
            'gt_json': np.load('Embeddings/embeddings_gt_json.npy'),
            'lx_json': np.load('Embeddings/embeddings_lx_json.npy'),
        }

        sims = {
            'text_vs_lx': np.load('Similarities/text_vs_lx.npy'),
            'text_vs_gt': np.load('Similarities/text_vs_gt.npy'),
            'gt_vs_lx': np.load('Similarities/gt_vs_lx.npy'),
        }
    else:
        os.makedirs('Embeddings', exist_ok=True)
        os.makedirs('Similarities', exist_ok=True)

        texts = df['text'].astype(str).tolist()
        gt_json_answers = df['gt_json_answer'].astype(str).tolist()
        lx_json_answers = df['lx_json_answer'].astype(str).tolist()

        embeddings = {
            'text': batch_embed(texts, 32, desc='Embeddings (TEXT)'),
            'gt_json': batch_embed(gt_json_answers, 32, desc='Embeddings (GT JSON)'),
            'lx_json': batch_embed(lx_json_answers, 32, desc='Embeddings (LX JSON)'),
        }

        np.save('Embeddings/embeddings_text.npy', embeddings['text'])
        np.save('Embeddings/embeddings_gt_json.npy', embeddings['gt_json'])
        np.save('Embeddings/embeddings_lx_json.npy', embeddings['lx_json'])

        sims = {
            'text_vs_gt': np.sum(embeddings['text'] * embeddings['gt_json'], axis=1),
            'text_vs_lx': np.sum(embeddings['text'] * embeddings['lx_json'], axis=1),
            'gt_vs_lx': np.sum(embeddings['gt_json'] * embeddings['lx_json'], axis=1),
        }

        np.save('Similarities/text_vs_gt.npy', sims['text_vs_gt'])
        np.save('Similarities/text_vs_lx.npy', sims['text_vs_lx'])
        np.save('Similarities/gt_vs_lx.npy', sims['gt_vs_lx'])

    # semantic_results = {'embeddings': embeddings, 'similarities': sims}

    df['text_vs_gt_similarity_score'] = sims['text_vs_gt']
    df['text_vs_lx_similarity_score'] = sims['text_vs_lx']
    df['gt_vs_lx_similarity_score'] = sims['gt_vs_lx']

    return df


def normalize_json(json_str: str) -> str:
    def json_dump(obj: json.JSONEncoder) -> str:
        # sort_keys=True
        return json.dumps(obj, ensure_ascii=False)
    if json_str is None:
        return None

    s = str(json_str).strip()

    try:  # 1) proper JSON
        return json_dump(json.loads(s))
    except Exception:
        pass

    try:  # 2) python literal (single quotes, None, etc.)
        return json_dump(ast.literal_eval(s))
    except Exception:
        pass

    try:  # 3) small repairs: convert JSON null->None and try again
        repaired = s.replace("null", "None")
        return json_dump(ast.literal_eval(repaired))
    except Exception:
        pass

    return s  # 4) last resort: return original (as safe string)


def join_gt_and_our_dataframes(gt_df_path: str, our_df_path: str) -> pd.DataFrame:
    """ Joins the ground truth dataframe with our dataframe on 'unique_id' column. """
    gt_df = pd.read_parquet(gt_df_path)
    our_df = pd.read_parquet(our_df_path)

    # sort both dataframes by 'unique_id' to ensure alignment
    gt_df = gt_df.sort_values(by='unique_id').reset_index(drop=True)
    our_df = our_df.sort_values(by='unique_id').reset_index(drop=True)

    # drop columns: ['candidate_example', 'candidate_text', 'values_indices', 'attributes', 'attributes_values', 'values']
    cols_to_drop = [
        'candidate_example',
        'candidate_text',
        'values_indices',
        'attributes',
        'attributes_values',
        'values',
        'id',
        'category',
        'source',
        'split',
    ]

    gt_df = gt_df.drop(columns=cols_to_drop)
    our_df = our_df.drop(columns=cols_to_drop)

    # rename 'json_answer' in gt_df to 'gt_json_answer'; rename 'json_answer' in our_df to 'lx_json_answer'
    gt_df = gt_df.rename(columns={'json_answer': 'gt_json_answer'})
    our_df = our_df.rename(columns={'json_answer': 'lx_json_answer'})

    # For each unique_id in gt_df, find the corresponding row in our_df and store the 'lx_json_answer' in a new column 'lx_json_answer' in gt_df
    merged_df = pd.merge(
        gt_df, our_df[['unique_id', 'lx_json_answer']], on='unique_id', how='left'
    )

    # Drop similarity_score column
    merged_df = merged_df.drop(columns=['similarity_score'])

    # Add columns: 'gt_similarity_score' and 'lx_similarity_score' with default value NA
    # merged_df['text_vs_gt_similarity_score'] = pd.NA
    # merged_df['text_vs_lx_similarity_score'] = pd.NA
    # merged_df['gt_vs_lx_similarity_score'] = pd.NA

    # Sort merged_df columns by the following order
    columns_order = [
        'unique_id',
        'dataset',
        'text',
        'gt_json_answer',
        'lx_json_answer',
        # 'text_vs_gt_similarity_score',
        # 'text_vs_lx_similarity_score',
        # 'gt_vs_lx_similarity_score',
    ]
    merged_df = merged_df[columns_order]

    # Drop all rows where 'lx_json_answer' is NA
    merged_df = merged_df.dropna(subset=['lx_json_answer'])

    # Normalize JSON answers for better comparison
    merged_df['gt_json_answer'] = merged_df['gt_json_answer'].apply(
        normalize_json)
    merged_df['lx_json_answer'] = merged_df['lx_json_answer'].apply(
        normalize_json)

    return merged_df
