""" Module for evaluating model solutions and plotting results. """

from typing import Dict


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
        model_keys = set(k.lower() for k in model_solutions.keys())
        reference_keys = set(k.lower() for k in reference_solutions.keys())
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
        # 'accuracy': 0.0,
        'precision': precision,
        'recall': recall,
        'f1': get_f1(precision, recall),
    }

    return scores


def plot_results() -> None:
    """ Plots the evaluation results """
    pass