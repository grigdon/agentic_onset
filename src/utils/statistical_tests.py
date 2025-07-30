import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def load_prediction_data(file_path):
    """
    Loads prediction data (true labels and probabilities) from a CSV file.

    Args:
        file_path (str): Path to the CSV file containing 'true_labels' and 'predictions' columns.

    Returns:
        tuple: A tuple containing:
            - np.array: True labels.
            - np.array: Predicted probabilities.
            Returns (None, None) if an error occurs.
    """
    try:
        df = pd.read_csv(file_path)
        return df['true_labels'].values, df['predictions'].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def delong_test_proc_style(y_true, y_pred1, y_pred2, debug_label=None):
    """
    Performs the DeLong test for comparing two ROC curves,
    following the pROC R package's implementation logic.

    Args:
        y_true (np.array): True binary labels.
        y_pred1 (np.array): Predicted probabilities from the first model.
        y_pred2 (np.array): Predicted probabilities from the second model.
        debug_label (str, optional): Label for debug output. Defaults to None.

    Returns:
        tuple: (z_statistic, two-sided p-value, one-sided p-value, AUC1, AUC2).
               Returns (None, None, None, None, None) if an error or numerical issue occurs.
    """
    try:
        fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
        fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
        auc1 = auc(fpr1, tpr1)
        auc2 = auc(fpr2, tpr2)

        auc_diff = auc1 - auc2

        n = len(y_true)
        n_pos = np.sum(y_true == 1)
        n_neg = n - n_pos

        v10 = np.zeros(n)
        v01 = np.zeros(n)

        for i in range(n):
            if y_true[i] == 1:  # Positive case
                v10[i] = np.sum(y_pred1[y_true == 0] < y_pred1[i]) / n_neg
                v01[i] = np.sum(y_pred2[y_true == 0] < y_pred2[i]) / n_neg
            else:  # Negative case
                v10[i] = np.sum(y_pred1[y_true == 1] > y_pred1[i]) / n_pos
                v01[i] = np.sum(y_pred2[y_true == 1] > y_pred2[i]) / n_pos

        v_diff = v10 - v01
        var_diff = np.var(v_diff) / n

        if debug_label == 'base':
            print(f"[DEBUG - base] auc_diff: {auc_diff}")
            print(f"[DEBUG - base] var_diff: {var_diff}")

        if var_diff <= 0 or np.isnan(var_diff) or np.isinf(var_diff):
            print(f"Warning: Non-positive or invalid variance detected: {var_diff}")
            return None, None, None, None, None

        z_stat = auc_diff / np.sqrt(var_diff)

        if abs(z_stat) > 100:
            print(f"Warning: Extreme z-statistic detected: {z_stat}")

        p_value_two_sided = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        p_value_one_sided = 1 - stats.norm.cdf(z_stat)

        # Handle numerical precision for p-values
        if p_value_two_sided < 1e-16: p_value_two_sided = 0.0
        if p_value_one_sided < 1e-16: p_value_one_sided = 0.0
        if p_value_one_sided > 1 - 1e-16: p_value_one_sided = 1.0

        return z_stat, p_value_two_sided, p_value_one_sided, auc1, auc2

    except Exception as e:
        print(f"Error in pROC-style Delong test: {e}")
        return None, None, None, None, None

def perform_delong_comparison(y_true, y_pred1, y_pred2, model1_name, model2_name, debug_label=None):
    """
    Performs a DeLong test comparison between two models and formats the results.
    Assumes model1 is the 'AI' model and model2 is the 'Human' model for one-sided testing.

    Args:
        y_true (np.array): True binary labels.
        y_pred1 (np.array): Predicted probabilities from the first model.
        y_pred2 (np.array): Predicted probabilities from the second model.
        model1_name (str): Name of the first model.
        model2_name (str): Name of the second model.
        debug_label (str, optional): Label for debug output. Defaults to None.

    Returns:
        dict: Dictionary containing comparison results, or None if an error occurs.
    """
    try:
        fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
        fpr2, tpr2, _ = roc_curve(y_true, y_pred2)

        auc1 = auc(fpr1, tpr1)
        auc2 = auc(fpr2, tpr2)

        z_score, p_value_two_sided, p_value_one_sided, _, _ = delong_test_proc_style(y_true, y_pred1, y_pred2, debug_label=debug_label)

        auc_diff = auc1 - auc2 # Model1 AUC - Model2 AUC

        significant = p_value_one_sided < 0.05 if p_value_one_sided is not None else False

        ci_lower = ci_upper = auc_diff
        if z_score is not None and z_score != 0:
            se_diff = abs(auc_diff / z_score)
            ci_lower = auc_diff - 1.96 * se_diff
            ci_upper = auc_diff + 1.96 * se_diff

        return {
            'model1_name': model1_name,
            'model2_name': model2_name,
            'auc1': auc1,
            'auc2': auc2,
            'auc_diff': auc_diff,
            'delong_statistic': z_score,
            'p_value_two_sided': p_value_two_sided,
            'p_value_one_sided': p_value_one_sided,
            'significant': significant,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'fpr1': fpr1,
            'tpr1': tpr1,
            'fpr2': fpr2,
            'tpr2': tpr2
        }
    except Exception as e:
        print(f"Error in DeLong comparison for {model1_name} vs {model2_name}: {e}")
        return None
