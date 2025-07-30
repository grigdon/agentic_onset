import json
import re
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data_processing.preprocessor import create_preprocessor

def validate_parameters(params):
    """
    Validates if the suggested parameters are valid for RandomForestClassifier.

    Args:
        params (dict): Dictionary of suggested parameters.

    Returns:
        bool: True if parameters are valid, False otherwise.
    """
    valid = True
    errors = []

    valid_rf_params = list(RandomForestClassifier().get_params().keys())
    for key in params.keys():
        if key not in valid_rf_params:
            errors.append(f"Bad parameter name: '{key}'")
            valid = False

    if 'n_estimators' in params and (not isinstance(params['n_estimators'], int) or params['n_estimators'] < 1):
        errors.append(f"n_estimators ({params['n_estimators']}) must be a positive whole number.")
        valid = False

    if 'max_depth' in params and params['max_depth'] is not None and (not isinstance(params['max_depth'], int) or params['max_depth'] < 1):
        errors.append(f"max_depth ({params['max_depth']}) must be a positive whole number or None.")
        valid = False

    if 'min_samples_split' in params and (not isinstance(params['min_samples_split'], int) or params['min_samples_split'] < 2):
        errors.append(f"min_samples_split ({params['min_samples_split']}) must be a whole number 2 or greater.")
        valid = False

    if 'min_samples_leaf' in params and (not isinstance(params['min_samples_leaf'], int) or params['min_samples_leaf'] < 1):
        errors.append(f"min_samples_leaf ({params['min_samples_leaf']}) must be a whole number 1 or greater.")
        valid = False

    if 'max_features' in params:
        mf = params['max_features']
        if isinstance(mf, str):
            if mf not in ['sqrt', 'log2', None]:
                errors.append(f"max_features string '{mf}' not valid (use 'sqrt', 'log2', or null).")
                valid = False
        elif isinstance(mf, (int, float)):
            if mf <= 0:
                errors.append(f"max_features numeric ({mf}) must be positive.")
                valid = False
            if isinstance(mf, float) and not (0.0 < mf <= 1.0):
                errors.append(f"max_features float ({mf}) must be between 0.0 and 1.0.")
                valid = False
        else:
            errors.append(f"max_features '{mf}' has a weird type.")
            valid = False

    if 'bootstrap' in params and not isinstance(params['bootstrap'], bool):
        errors.append(f"bootstrap ({params['bootstrap']}) must be true or false.")
        valid = False

    if 'class_weight' in params:
        cw = params['class_weight']
        if not (cw is None or (isinstance(cw, str) and cw in ['balanced', 'balanced_subsample'])):
            errors.append(f"class_weight '{cw}' not valid (use 'balanced', 'balanced_subsample', or null).")
            valid = False

    if not valid:
        print(f"Validation Fails: {'; '.join(errors)}")
    return valid

def get_baseline_prompt(theory, stage, baseline_auc, human_params, history, theory_description, best_ai_val_auc):
    """
    Generates the prompt for the LLM to suggest Random Forest hyperparameters.

    Args:
        theory (str): The name of the theoretical model.
        stage (str): The prediction stage ('onset' or 'escalation').
        baseline_auc (float): The human-tuned model's test AUC.
        human_params (dict): Fixed hyperparameters from the human baseline.
        history (str): JSON string of previous AI optimization attempts.
        theory_description (str): Description of the theoretical model.
        best_ai_val_auc (float): The best validation AUC achieved by the AI so far.

    Returns:
        str: The formatted prompt for the LLM.
    """
    stage_description = "early mobilization patterns" if stage == "onset" else "escalation from nonviolent to violent"
    display_human_params = {k: v for k, v in human_params.items() if k not in ['oob_score', 'random_state', 'n_jobs']}

    return f"""You are an expert in Random Forest hyperparameter optimization for political conflict prediction.

Current HUMAN-TUNED baseline performance (from 10-fold cross-validation with downsampling and mtry tuning, evaluated on a held-out test set):
- Model: {theory} ({stage} prediction)
- Baseline Test AUC (Human-Tuned): {baseline_auc:.4f}
- Human Fixed Parameters (mtry/max_features was tuned): {display_human_params}

Previous AI attempts (Validation AUCs from internal validation set): {history}

Your goal: Suggest ONE new parameter set for the RandomForestClassifier to improve upon the human baseline. You will be optimizing against a separate validation set.

Key considerations:
1. This is {stage} prediction - {stage_description}
2. Theory focus: {theory_description}
3. Dataset characteristics: Imbalanced classes. Remember that the human baseline already uses a form of balancing (downsampling). Your model's `class_weight` parameter is a way to handle this.
4. Current best AI Validation AUC achieved so far in previous iterations: {best_ai_val_auc:.4f}

Respond ONLY with valid JSON, ensuring all keys are valid RandomForestClassifier constructor arguments:
{{"n_estimators": int, "max_depth": int|null, "min_samples_split": int,
  "min_samples_leaf": int, "max_features": "sqrt"|"log2"|float,
  "bootstrap": true|false, "class_weight": "balanced"|"balanced_subsample"|null}}"""

def llm_hyperparameter_optimization(client, X_train_ai, y_train_ai, X_val_ai, y_val_ai, features,
                                  theory, stage, human_test_auc, fixed_params, openai_model, max_iterations,
                                  numeric_features, categorical_features, theory_descriptions):
    """
    Orchestrates the LLM-driven hyperparameter optimization process.

    Args:
        client (openai.OpenAI): OpenAI API client instance.
        X_train_ai (pd.DataFrame): Training features for AI optimization.
        y_train_ai (pd.Series): Training target labels for AI optimization.
        X_val_ai (pd.DataFrame): Validation features for AI optimization.
        y_val_ai (pd.Series): Validation target labels for AI optimization.
        features (list): List of features used for this specific model.
        theory (str): The name of the theoretical model.
        stage (str): The prediction stage ('onset' or 'escalation').
        human_test_auc (float): The human-tuned model's test AUC (for comparison).
        fixed_params (dict): Fixed Random Forest hyperparameters.
        openai_model (str): Name of the OpenAI model to use.
        max_iterations (int): Maximum number of LLM optimization iterations.
        numeric_features (list): Global list of all numeric features.
        categorical_features (list): Global list of all categorical features.
        theory_descriptions (dict): Dictionary of theoretical model descriptions.

    Returns:
        tuple: A tuple containing:
            - dict: Best hyperparameters found by the LLM.
            - list: History of all LLM optimization trials.
            - float: Best validation AUC achieved by the LLM.
    """
    preprocessor = create_preprocessor(features, numeric_features, categorical_features)
    trial_history = []
    best_ai_val_auc = 0.0
    best_params_found = fixed_params.copy()

    theory_desc = theory_descriptions.get(theory, "An unclassified theoretical model")

    for iteration in tqdm(range(max_iterations), desc=f"LLM Optimizing {theory} {stage}"):
        prompt = get_baseline_prompt(
            theory=theory,
            stage=stage,
            baseline_auc=human_test_auc,
            human_params=fixed_params,
            history=json.dumps(trial_history[-3:], indent=2) if trial_history else "No previous attempts.",
            theory_description=theory_desc,
            best_ai_val_auc=best_ai_val_auc
        )

        response = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        content = response.choices[0].message.content

        json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
        if not json_match:
            print(f"  Iteration {iteration+1}: No valid JSON found in LLM response. Skipping.")
            continue

        try:
            suggested_params = json.loads(json_match.group(0))
        except json.JSONDecodeError as e:
            print(f"  Iteration {iteration+1}: Invalid JSON format from LLM: {e}. Skipping.")
            continue

        if not validate_parameters(suggested_params):
            print(f"  Iteration {iteration+1}: LLM suggested invalid parameters: {suggested_params}. Skipping.")
            continue

        current_iter_params = fixed_params.copy()
        current_iter_params.update(suggested_params)

        if current_iter_params.get('bootstrap', True) is False:
            current_iter_params['oob_score'] = False

        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**current_iter_params))
        ])

        model.fit(X_train_ai, y_train_ai)
        val_preds = model.predict_proba(X_val_ai)[:, 1]
        current_val_auc = roc_auc_score(y_val_ai, val_preds)

        trial_record = {
            "iteration": iteration + 1,
            "params": suggested_params,
            "applied_params": current_iter_params,
            "validation_auc": current_val_auc,
            "beats_human_baseline_test_auc": bool(current_val_auc > human_test_auc)
        }
        trial_history.append(trial_record)

        if current_val_auc > best_ai_val_auc:
            best_ai_val_auc = current_val_auc
            best_params_found = current_iter_params.copy()

        print(f"  Iteration {iteration+1}: Val AUC={current_val_auc:.4f} (Best so far: {best_ai_val_auc:.4f}) {'(Improved!)' if current_val_auc > best_ai_val_auc else ''}")

    return best_params_found, trial_history, best_ai_val_auc
