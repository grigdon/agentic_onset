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
    # This function remains unchanged
    valid = True
    errors = []
    valid_rf_params = list(RandomForestClassifier().get_params().keys())
    for key in params.keys():
        if key not in valid_rf_params:
            errors.append(f"Bad parameter name: '{key}'")
            valid = False
    if not valid:
        print(f"Validation Fails: {'; '.join(errors)}")
    return valid

def get_baseline_prompt(theory, stage, baseline_auc, human_params, history, theory_description, best_ai_val_auc, prompt_variation='original'):
    """
    Generates the prompt for the LLM based on the specified variation.
    """
    # --- Original Prompt (Detailed) ---
    if prompt_variation == 'original':
        stage_description = "early mobilization patterns" if stage == "onset" else "escalation from nonviolent to violent"
        display_human_params = {k: v for k, v in human_params.items() if k not in ['oob_score', 'random_state', 'n_jobs']}
        return f"""You are an expert in Random Forest hyperparameter optimization for political conflict prediction.

Current HUMAN-TUNED baseline performance:
- Model: {theory} ({stage} prediction)
- Baseline Test AUC (Human-Tuned): {baseline_auc:.4f}
- Human Fixed Parameters: {display_human_params}

Previous AI attempts (Validation AUCs): {history}
Current best AI Validation AUC: {best_ai_val_auc:.4f}

Your goal: Suggest ONE new parameter set for RandomForestClassifier to improve performance.

Key considerations:
1. This is {stage} prediction - {stage_description}.
2. Theory focus: {theory_description}.
3. Dataset is imbalanced. `class_weight` is a key parameter.

Respond ONLY with valid JSON:
{{"n_estimators": int, "max_depth": int|null, "min_samples_split": int, "min_samples_leaf": int, "max_features": "sqrt"|"log2"|float, "bootstrap": true|false, "class_weight": "balanced"|"balanced_subsample"|null}}"""

    # --- Minimalist Prompt ---
    elif prompt_variation == 'minimalist':
        return f"""You are a Random Forest tuning expert.
Previous attempts (Validation AUCs): {history}
Suggest ONE new set of hyperparameters for RandomForestClassifier.
Respond ONLY with valid JSON.
{{"n_estimators": int, "max_depth": int|null, "min_samples_split": int, "min_samples_leaf": int, "max_features": "sqrt"|"log2"|float, "bootstrap": true|false, "class_weight": "balanced"|"balanced_subsample"|null}}"""

    # --- Prescriptive Prompt ---
    elif prompt_variation == 'prescriptive':
        return f"""You are an expert tuning RandomForestClassifier for an imbalanced dataset.
Your primary goal is to find parameters that generalize well and avoid overfitting.

Previous attempts (Validation AUCs): {history}

Suggest ONE new parameter set. Focus on regularization by adjusting `max_depth`, `min_samples_split`, and `min_samples_leaf`. Also, consider using `class_weight` to handle imbalance. Avoid excessively large `n_estimators`.

Respond ONLY with valid JSON:
{{"n_estimators": int, "max_depth": int|null, "min_samples_split": int, "min_samples_leaf": int, "max_features": "sqrt"|"log2"|float, "bootstrap": true|false, "class_weight": "balanced"|"balanced_subsample"|null}}"""
    
    else:
        raise ValueError(f"Unknown prompt variation: {prompt_variation}")


def llm_hyperparameter_optimization(client, X_train_ai, y_train_ai, X_val_ai, y_val_ai, features,
                                  theory, stage, human_test_auc, fixed_params, openai_model, max_iterations,
                                  numeric_features, categorical_features, theory_descriptions,
                                  prompt_variation='original'):
    """
    Orchestrates the LLM-driven hyperparameter optimization process.
    """
    preprocessor = create_preprocessor(features, numeric_features, categorical_features)
    trial_history = []
    best_ai_val_auc = 0.0
    best_params_found = fixed_params.copy()
    theory_desc = theory_descriptions.get(theory, "An unclassified theoretical model")

    for iteration in tqdm(range(max_iterations), desc=f"LLM Optimizing ({prompt_variation})"):
        prompt = get_baseline_prompt(
            theory=theory, stage=stage, baseline_auc=human_test_auc,
            human_params=fixed_params, history=json.dumps(trial_history[-3:], indent=2) if trial_history else "No previous attempts.",
            theory_description=theory_desc, best_ai_val_auc=best_ai_val_auc,
            prompt_variation=prompt_variation
        )
        response = client.chat.completions.create(model=openai_model, messages=[{"role": "user", "content": prompt}], temperature=0.7)
        content = response.choices[0].message.content
        try:
            suggested_params = json.loads(content)
        except json.JSONDecodeError: continue
        if not validate_parameters(suggested_params): continue
        current_iter_params = fixed_params.copy(); current_iter_params.update(suggested_params)
        model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**current_iter_params))])
        model.fit(X_train_ai, y_train_ai)
        val_preds = model.predict_proba(X_val_ai)[:, 1]
        current_val_auc = roc_auc_score(y_val_ai, val_preds)
        trial_history.append({"iteration": iteration + 1, "params": suggested_params, "validation_auc": current_val_auc})
        if current_val_auc > best_ai_val_auc:
            best_ai_val_auc = current_val_auc
            best_params_found = current_iter_params.copy()

    return best_params_found, trial_history, best_ai_val_auc