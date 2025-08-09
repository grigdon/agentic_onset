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

def _compute_midrank(x):
    """
    Computes midrank for tied values in an array.

    Args:
        x (np.array): Input array.

    Returns:
        np.array: Array with midranks.
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T + 1
    return T2

def _compute_ground_truth_statistics(true):
    """
    Sorts ground truth labels and counts positive samples.

    Args:
        true (np.array): True binary labels.

    Returns:
        tuple: (order, label_1_count)
            - order (np.array): Indices that sort the true labels.
            - label_1_count (int): Number of positive labels.
    """
    assert np.array_equal(np.unique(true), [0, 1]), "Ground truth must be binary."
    order = (-true).argsort()
    label_1_count = int(true.sum())
    return order, label_1_count

def delong_test_fast(y_true, y_pred1, y_pred2, debug_label=None):
    """
    Performs DeLong's test for comparing the AUCs of two models
    using a fast, vectorized implementation.

    Args:
        y_true (np.array): True binary labels.
        y_pred1 (np.array): Predicted probabilities from the first model.
        y_pred2 (np.array): Predicted probabilities from the second model.
        debug_label (str, optional): Label for debug output. Defaults to None.

    Returns:
        tuple: (z_statistic, two-sided p-value, one-sided p-value, AUC1, AUC2).
               Returns (None, None, None, None, None) if an error occurs.
    """
    try:
        # Prepare data
        order, label_1_count = _compute_ground_truth_statistics(y_true)
        probs_A = y_pred1[order]
        probs_B = y_pred2[order]
        
        # Fast DeLong computation starts here
        m = label_1_count  # Number of positive samples
        n = len(y_true) - m  # Number of negative samples
        
        if m == 0 or n == 0:
            print("Warning: One class has zero samples. Cannot perform DeLong's test.")
            return None, None, None, None, None
            
        positive_probs_A = probs_A[:m]
        negative_probs_A = probs_A[m:]
        positive_probs_B = probs_B[:m]
        negative_probs_B = probs_B[m:]

        # Midrank computations
        txA = _compute_midrank(positive_probs_A)
        tyA = _compute_midrank(negative_probs_A)
        tzA = _compute_midrank(probs_A)
        txB = _compute_midrank(positive_probs_B)
        tyB = _compute_midrank(negative_probs_B)
        tzB = _compute_midrank(probs_B)

        # Calculate AUCs
        auc1 = tzA[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
        auc2 = tzB[:m].sum() / (m * n) - (m + 1.0) / (2.0 * n)
        
        # Compute variance components
        v01A = (tzA[:m] - txA) / n
        v10A = 1.0 - (tzA[m:] - tyA) / m
        v01B = (tzB[:m] - txB) / n
        v10B = 1.0 - (tzB[m:] - tyB) / m
        
        # Construct covariance matrix inputs
        v01_stack = np.vstack((v01A, v01B))
        v10_stack = np.vstack((v10A, v10B))

        # Compute covariance matrices
        sx = np.cov(v01_stack)
        sy = np.cov(v10_stack)
        delongcov = sx / m + sy / n

        # Calculating z-score and p-value
        l = np.array([[1, -1]])
        auc_diff = auc1 - auc2
        var_diff = np.dot(np.dot(l, delongcov), l.T).flatten()[0]
        
        if debug_label == 'base':
            print(f"[DEBUG - base] auc_diff: {auc_diff}")
            print(f"[DEBUG - base] var_diff: {var_diff}")
        
        if var_diff <= 0 or np.isnan(var_diff):
            print(f"Warning: Non-positive or invalid variance detected: {var_diff}")
            return None, None, None, None, None

        z_stat = auc_diff / np.sqrt(var_diff)

        if abs(z_stat) > 100:
            print(f"Warning: Extreme z-statistic detected: {z_stat}")

        p_value_two_sided = stats.norm.sf(abs(z_stat)) * 2
        p_value_one_sided = stats.norm.sf(z_stat)

        # Handle numerical precision for p-values
        if p_value_two_sided < 1e-16: p_value_two_sided = 0.0
        if p_value_one_sided < 1e-16: p_value_one_sided = 0.0
        if p_value_one_sided > 1 - 1e-16: p_value_one_sided = 1.0

        return z_stat, p_value_two_sided, p_value_one_sided, auc1, auc2

    except Exception as e:
        print(f"Error in fast Delong test: {e}")
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
        # Use the fast implementation
        z_score, p_value_two_sided, p_value_one_sided, auc1, auc2 = delong_test_fast(y_true, y_pred1, y_pred2, debug_label=debug_label)

        if z_score is None:
            return None

        auc_diff = auc1 - auc2 # Model1 AUC - Model2 AUC

        significant = p_value_one_sided < 0.05 if p_value_one_sided is not None else False

        ci_lower = ci_upper = auc_diff
        if z_score is not None and z_score != 0:
            se_diff = abs(auc_diff / z_score)
            ci_lower = auc_diff - 1.96 * se_diff
            ci_upper = auc_diff + 1.96 * se_diff

        fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
        fpr2, tpr2, _ = roc_curve(y_true, y_pred2)

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