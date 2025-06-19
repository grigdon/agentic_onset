import pandas as pd
import numpy as np
import json
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict
from datetime import datetime

# --- Mocking sklearn and OpenAI components ---
# We'll replace actual computations with dummy data generation
class MockRandomForestClassifier:
    def __init__(self, **kwargs):
        self.feature_importances_ = np.random.rand(10) # Dummy importance
        self.feature_importances_ /= self.feature_importances_.sum() # Normalize

    def fit(self, X, y):
        pass # Do nothing
    
    def predict_proba(self, X):
        # Generate dummy probabilities
        return np.column_stack([1 - np.random.rand(X.shape[0]) * 0.2, np.random.rand(X.shape[0]) * 0.2 + 0.5])

class MockPipeline:
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: obj for name, obj in steps}
    
    def fit(self, X, y):
        pass # Do nothing

    def predict_proba(self, X):
        return self.named_steps['classifier'].predict_proba(X)

class MockColumnTransformer:
    def __init__(self, transformers, remainder='drop'):
        self.transformers_ = transformers
        self.remainder = remainder
    
    def fit(self, X):
        pass # Do nothing
    
    def transform(self, X):
        return X # Return X as is for simplicity in mock

    def get_feature_names_out(self, input_features=None):
        # Generate some dummy feature names based on expected structure
        names = []
        for name, transformer, features_in_transformer in self.transformers_:
            if name == 'num':
                names.extend(features_in_transformer)
            elif name == 'cat':
                # For categorical, simulate one-hot encoding by adding suffixes
                for feat in features_in_transformer:
                    names.append(f"{feat}_cat1")
                    names.append(f"{feat}_cat2") # Simulate a few categories
        return names

# --- Mocking sklearn and OpenAI components ---
# We'll replace actual computations with dummy data generation
class MockRandomForestClassifier:
    def __init__(self, **kwargs):
        self.feature_importances_ = np.random.rand(10) # Dummy importance
        self.feature_importances_ /= self.feature_importances_.sum() # Normalize

    def fit(self, X, y):
        pass # Do nothing
    
    def predict_proba(self, X):
        # Generate dummy probabilities
        return np.column_stack([1 - np.random.rand(X.shape[0]) * 0.2, np.random.rand(X.shape[0]) * 0.2 + 0.5])

class MockPipeline:
    def __init__(self, steps):
        self.steps = steps
        self.named_steps = {name: obj for name, obj in steps}
    
    def fit(self, X, y):
        pass # Do nothing

    def predict_proba(self, X):
        return self.named_steps['classifier'].predict_proba(X)

class MockColumnTransformer:
    def __init__(self, transformers, remainder='drop'):
        self.transformers_ = transformers
        self.remainder = remainder
    
    def fit(self, X):
        pass # Do nothing
    
    def transform(self, X):
        return X # Return X as is for simplicity in mock

    def get_feature_names_out(self, input_features=None):
        # Generate some dummy feature names based on expected structure
        names = []
        for name, transformer, features_in_transformer in self.transformers_:
            if name == 'num':
                names.extend(features_in_transformer)
            elif name == 'cat':
                # For categorical, simulate one-hot encoding by adding suffixes
                for feat in features_in_transformer:
                    names.append(f"{feat}_cat1")
                    names.append(f"{feat}_cat2") # Simulate a few categories
        return names

# FIX: Corrected MockOpenAIClient structure
class MockOpenAIClient:
    def __init__(self, api_key):
        self._chat = self._MockChat()
        self._models = self._MockModels() # Initialize MockModels

    class _MockChatCompletions:
        def create(self, model, messages, temperature):
            # Simulate LLM response with reasonable parameters
            n_est = np.random.choice([500, 1000, 1500])
            max_d = np.random.choice([10, 20, None])
            min_s_split = np.random.choice([2, 5])
            min_s_leaf = np.random.choice([1, 2])
            max_f = np.random.choice(['sqrt', 'log2', 0.8, 1.0])
            bootstrap = np.random.choice([True, False])
            class_w = np.random.choice(['balanced', None])

            response_content = json.dumps({
                "n_estimators": int(n_est),
                "max_depth": max_d,
                "min_samples_split": int(min_s_split),
                "min_samples_leaf": int(min_s_leaf),
                "max_features": max_f,
                "bootstrap": bool(bootstrap),
                "class_weight": class_w
            })
            
            class MockChoice:
                def __init__(self, content):
                    self.message = type('obj', (object,), {'content': content})()
            
            class MockResponse:
                def __init__(self, content):
                    self.choices = [MockChoice(content)]
            
            return MockResponse(response_content)

    class _MockChat:
        def __init__(self):
            self.completions = self._MockChatCompletions()

    class _MockModels:
        def list(self):
            pass # Simulate successful API key check

    @property
    def chat(self):
        return self._chat

    @property
    def models(self):
        return self._models

# Rest of the script remains the same...

# =====================================================================
# CONFIGURATION - EXACT MATCH TO R SCRIPT (Human Baseline adjustments for replication)
# =====================================================================

# Exact theoretical models from R script, confirmed to match R formulas.
THEORETICAL_MODELS = {
    "base": {
        "onset": ["t_claim", "coldwar", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "coldwar", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    },
    "complete": {
        "onset": ["t_claim", "lv2x_polyarchy", "lfederal", "regaut", "status_excl", "lost_autonomy", 
                 "downgr2_aut", "lsepkin_adjregbase1", "lgiantoilfield", "mounterr", "coldwar", 
                 "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "lv2x_polyarchy", "lfederal", "regaut", "status_excl", "lost_autonomy", 
                      "downgr2_aut", "lsepkin_adjregbase1", "lgiantoilfield", "mounterr", "coldwar", 
                      "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    },
    "resource_mobilization": {
        "onset": ["t_claim", "lsepkin_adjregbase1", "lgiantoilfield", "mounterr", "coldwar", 
                 "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "lsepkin_adjregbase1", "lgiantoilfield", "mounterr", "coldwar", 
                      "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    },
    "political_opportunity": {
        "onset": ["t_claim", "lv2x_polyarchy", "lfederal", "regaut", "coldwar", "noncontiguous", 
                 "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "lv2x_polyarchy", "lfederal", "regaut", "coldwar", "noncontiguous", 
                      "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    },
    "grievances": {
        "onset": ["t_claim", "status_excl", "downgr2_aut", "lost_autonomy", "coldwar", "noncontiguous", 
                 "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "status_excl", "downgr2_aut", "lost_autonomy", "coldwar", "noncontiguous", 
                      "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    },
    "pom_gm": {
        "onset": ["t_claim", "lv2x_polyarchy", "lfederal", "regaut", "status_excl", "downgr2_aut", 
                 "lost_autonomy", "coldwar", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "lv2x_polyarchy", "lfederal", "regaut", "status_excl", "downgr2_aut", 
                      "lost_autonomy", "coldwar", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    },
    "pom_rmm": {
        "onset": ["t_claim", "lv2x_polyarchy", "lfederal", "coldwar", "regaut", "lsepkin_adjregbase1", 
                 "lgiantoilfield", "mounterr", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "lv2x_polyarchy", "lfederal", "regaut", "lsepkin_adjregbase1", 
                      "lgiantoilfield", "mounterr", "coldwar", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    },
    "rmm_gm": {
        "onset": ["t_claim", "lsepkin_adjregbase1", "lgiantoilfield", "mounterr", "status_excl", 
                 "downgr2_aut", "lost_autonomy", "coldwar", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"],
        "escalation": ["t_escal", "lsepkin_adjregbase1", "lgiantoilfield", "mounterr", "status_excl", 
                      "downgr2_aut", "lost_autonomy", "coldwar", "noncontiguous", "lnlrgdpcap", "lnltotpop", "groupsize", "numb_rel_grps"]
    }
}

HUMAN_RF_FIXED_PARAMS = {
    'n_estimators': 1000,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True,
    'random_state': 666,
    'n_jobs': -1,
    'oob_score': False
}

STAGE1_TARGET = "nviol_sdm_onset"
STAGE2_TARGET = "firstescal"

NUMERIC_FEATURES = ['groupsize', 'lsepkin_adjregbase1', 'lnlrgdpcap', 'lnltotpop', 
                   'lv2x_polyarchy', 'numb_rel_grps', 'mounterr', 't_claim', 't_escal']
CATEGORICAL_FEATURES = ['status_excl', 'lost_autonomy', 'downgr2_aut', 'regaut', 
                       'lgiantoilfield', 'noncontiguous', 'lfederal', 'coldwar']

OPENAI_MODEL = "gpt-4-turbo"
MAX_ITERATIONS = 5 # Reduced for quicker mock runs

# Enhanced prompts - FIXED VERSION
def get_baseline_prompt(theory, stage, baseline_auc, human_params, history, theory_description, best_ai_val_auc):
    """Generate the baseline prompt with proper string formatting."""
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

THEORY_DESCRIPTIONS = {
    "base": "Control variables only - demographic and temporal factors",
    "complete": "All theoretical mechanisms combined",
    "resource_mobilization": "Economic resources, terrain advantages, and mobilization capacity",
    "political_opportunity": "Political system openness, regime characteristics, and institutional factors",
    "grievances": "Group status, autonomy loss, and political exclusion",
    "pom_gm": "Political opportunities interacting with grievances",
    "pom_rmm": "Political opportunities enabling resource mobilization",
    "rmm_gm": "Resource availability amplifying grievances"
}

# =====================================================================
# MOCK DATA PROCESSING FUNCTIONS (NO ACTUAL DATA LOADING)
# =====================================================================

def mock_load_and_preprocess_data():
    """Mocks data loading, returns a dummy DataFrame."""
    print("\n=== MOCK: Loading and preprocessing data ===")
    # Create a dummy DataFrame with required columns and some rows
    # The actual values don't matter, just the structure for downstream functions
    num_rows = 1000
    data = {
        'isrelevant': np.ones(num_rows, dtype=int),
        'exclacc': np.zeros(num_rows, dtype=int),
        'gwgroupid': np.random.randint(1, 100, num_rows),
        'nviol_sdm_onset': np.random.choice([0, 1], num_rows, p=[0.8, 0.2]),
        'firstescal': np.random.choice([0, 1], num_rows, p=[0.7, 0.3])
    }
    
    # Add all theoretical model features with dummy data
    all_possible_features = set()
    for stage_theories in THEORETICAL_MODELS.values():
        for features_list in stage_theories.values():
            all_possible_features.update(features_list)

    for feature in all_possible_features:
        if feature not in data:
            if feature in NUMERIC_FEATURES:
                data[feature] = np.random.rand(num_rows) * 100
            elif feature in CATEGORICAL_FEATURES:
                data[feature] = np.random.randint(0, 3, num_rows) # Simulate 3 categories
    
    df = pd.DataFrame(data)
    print(f"MOCK: Raw data shape: {df.shape}")
    print("MOCK: Applying R script filters (isrelevant == 1, exclacc == 0)...")
    df_filtered = df[(df['isrelevant'] == 1) & (df['exclacc'] == 0)].copy()
    print(f"MOCK: Filtered data shape: {df_filtered.shape}")
    return df_filtered

def mock_prepare_stage_data(df, stage):
    """Mocks stage data preparation, returns a dummy DataFrame."""
    print(f"\n=== MOCK: Preparing {stage} data ===")
    target = STAGE1_TARGET if stage == "onset" else STAGE2_TARGET
    
    # Simulate NA dropping by reducing rows slightly
    df_stage = df.sample(frac=0.9).copy() 
    
    # Ensure target and gwgroupid are present and target has at least 2 unique values
    df_stage = df_stage[[target, 'gwgroupid'] + [f for f in df_stage.columns if f in NUMERIC_FEATURES or f in CATEGORICAL_FEATURES]].copy()

    if df_stage.empty:
        print(f"MOCK: Warning: {stage} stage DataFrame is empty after NA removal simulation.")
        return pd.DataFrame()

    # Ensure at least two unique target values
    if df_stage[target].nunique() < 2:
        # If not, force it for mock purposes
        if len(df_stage) > 10: # Ensure enough rows to create two classes
            df_stage.loc[df_stage.index[:5], target] = 0
            df_stage.loc[df_stage.index[5:10], target] = 1
        else:
            print(f"MOCK: Warning: Only one class present in target '{target}' for {stage} stage. Cannot train model.")
            return pd.DataFrame()

    print(f"MOCK: Stage data shape after NA drop simulation: {df_stage.shape}")
    print(f"MOCK: Target distribution for {stage}: {df_stage[target].value_counts().to_dict()}")
    
    return df_stage

def mock_group_aware_split(df, target_col, test_size=0.25, random_state=666):
    """Mocks group-aware splitting, returns dummy train/test DFs."""
    print(f"MOCK: Performing group-aware split with test_size={test_size} and random_state={random_state}...")
    
    # Simulate splitting by simply sampling rows
    train_df = df.sample(frac=1-test_size, random_state=random_state).copy()
    test_df = df.drop(train_df.index).copy()

    # Ensure test set is not empty and has both classes if possible
    if test_df.empty and len(df) > 1:
        test_df = df.sample(frac=0.1, random_state=random_state + 1).copy() # Ensure some test data
        train_df = df.drop(test_df.index).copy()

    if train_df.empty or test_df.empty or train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
        print(f"MOCK: Warning: Group split resulted in empty or single-class train/test set. Generating minimal valid split.")
        if len(df) > 20: # Ensure enough data for a minimal valid split
            # Crude way to ensure at least two classes
            train_df = df.sample(n=int(len(df)*0.75), random_state=random_state).copy()
            test_df = df.drop(train_df.index).copy()
            
            if train_df[target_col].nunique() < 2:
                train_df.loc[train_df.index[0], target_col] = 0
                train_df.loc[train_df.index[1], target_col] = 1
            if test_df[target_col].nunique() < 2:
                test_df.loc[test_df.index[0], target_col] = 0
                test_df.loc[test_df.index[1], target_col] = 1
        else:
            print("MOCK: Not enough data for a minimal valid split. Returning empty DataFrames.")
            return pd.DataFrame(), pd.DataFrame()

    print(f"MOCK: Train groups (simulated): {len(train_df['gwgroupid'].unique())}, Test groups (simulated): {len(test_df['gwgroupid'].unique())}")
    print(f"MOCK: Train samples: {len(train_df)} (Target distrib: {train_df[target_col].value_counts().to_dict()})")
    print(f"MOCK: Test samples: {len(test_df)} (Target distrib: {test_df[target_col].value_counts().to_dict()})")
    
    return train_df, test_df

def mock_create_preprocessor(features):
    """Mocks preprocessor creation."""
    numeric_features = [f for f in features if f in NUMERIC_FEATURES]
    categorical_features = [f for f in features if f in CATEGORICAL_FEATURES]
    
    transformers = []
    if numeric_features:
        transformers.append(('num', None, numeric_features)) # No actual transformer, just names
    if categorical_features:
        transformers.append(('cat', None, categorical_features))
    
    return MockColumnTransformer(transformers=transformers, remainder='drop') # Return mock object


# =====================================================================
# MOCK MODEL TRAINING AND EVALUATION FUNCTIONS
# =====================================================================

def mock_train_human_baseline_replication(X_train_full, y_train_full, groups_train_full, features):
    """Mocks training the human-tuned model."""
    print("\n--- MOCK: Training human-tuned baseline (R replication) ---")
    
    # Simulate downsampling by reducing majority class in y_train_full for mock
    mock_y_train = y_train_full.copy()
    if mock_y_train.nunique() > 1:
        majority_class = mock_y_train.value_counts().idxmax()
        minority_count = mock_y_train.value_counts().min()
        majority_indices = mock_y_train[mock_y_train == majority_class].index
        
        np.random.seed(HUMAN_RF_FIXED_PARAMS['random_state'])
        downsampled_majority_indices = np.random.choice(majority_indices, minority_count, replace=False)
        
        all_indices = np.concatenate([downsampled_majority_indices, mock_y_train[mock_y_train != majority_class].index])
        X_train_full = X_train_full.loc[all_indices].copy()
        mock_y_train = mock_y_train.loc[all_indices].copy()

    print(f"MOCK: Downsampled train data (simulated): {len(mock_y_train)} samples.")
    print(f"MOCK: Simulating GridSearchCV for mtry tuning. Best params found: {{'classifier__max_features': 'sqrt'}}")
    
    # Return a mock model with dummy data
    mock_model = MockPipeline([
        ('preprocessor', mock_create_preprocessor(features)),
        ('classifier', MockRandomForestClassifier())
    ])
    return mock_model

def mock_llm_hyperparameter_optimization(client, X_train_ai, y_train_ai, X_val_ai, y_val_ai, features, 
                                  theory, stage, human_test_auc):
    """Mocks LLM-guided hyperparameter optimization."""
    print(f"\n--- MOCK: Starting LLM optimization for {theory} ({stage}) ---")
    
    mock_preprocessor = mock_create_preprocessor(features)
    trial_history = []
    best_ai_val_auc = 0.70 # Starting mock best AUC
    best_params_found = HUMAN_RF_FIXED_PARAMS.copy()
    
    theory_desc = THEORY_DESCRIPTIONS.get(theory, "Unknown theory")
    
    for iteration in range(MAX_ITERATIONS): # Use range directly as tqdm not needed for mock
        print(f"MOCK LLM Optimizing {theory} {stage} - Iteration {iteration + 1}")
        # Simulate LLM call
        prompt = get_baseline_prompt(
            theory=theory,
            stage=stage,
            baseline_auc=human_test_auc,
            human_params=HUMAN_RF_FIXED_PARAMS,
            history=json.dumps(trial_history[-3:], indent=2) if trial_history else "None",
            theory_description=theory_desc,
            best_ai_val_auc=best_ai_val_auc
        )
        
        # Simulate response from mock OpenAI client
        response = client.chat.completions.create(
            model=OPENAI_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0.7
        )
        content = response.choices[0].message.content
        suggested_params = json.loads(content) # Assuming mock always returns valid JSON

        current_iter_params = HUMAN_RF_FIXED_PARAMS.copy()
        current_iter_params.update(suggested_params)
        
        # Simulate model training and evaluation
        mock_model = MockPipeline([
            ('preprocessor', mock_preprocessor),
            ('classifier', MockRandomForestClassifier())
        ])
        
        # Generate a mock validation AUC that might improve
        current_val_auc = np.random.uniform(max(0.65, best_ai_val_auc - 0.02), best_ai_val_auc + 0.03)
        current_val_auc = min(0.95, current_val_auc) # Cap at 0.95

        trial_record = {
            "iteration": iteration + 1,
            "params": suggested_params,
            "applied_params": current_iter_params,
            "validation_auc": current_val_auc,
            "beats_human_baseline_test_auc": current_val_auc > human_test_auc
        }
        trial_history.append(trial_record)
        
        if current_val_auc > best_ai_val_auc:
            best_ai_val_auc = current_val_auc
            best_params_found = current_iter_params.copy()
        
        print(f"  MOCK: Iteration {iteration+1}: Validation AUC={current_val_auc:.4f} (Best so far: {best_ai_val_auc:.4f}) {'(Improved)' if current_val_auc > best_ai_val_auc else ''}")
    
    print(f"\nMOCK: LLM Optimization for {theory} ({stage}) completed.")
    print(f"MOCK: Best validation AUC achieved during LLM search: {best_ai_val_auc:.4f}")
    print(f"MOCK: Best parameters found: {best_params_found}")
    return best_params_found, trial_history, best_ai_val_auc

def mock_evaluate_model(model, X_test, y_test, model_type_name, features_list):
    """Mocks comprehensive model evaluation."""
    print(f"MOCK: Evaluating {model_type_name} model on test set...")
    
    if X_test.empty or y_test.empty or y_test.nunique() < 2:
        print(f"MOCK: Warning: Insufficient data for {model_type_name} test evaluation or single class present. Returning default values.")
        return {
            'auc': 0.5,
            'predictions': np.array([]),
            'true_labels': np.array([]),
            'feature_importance': pd.DataFrame(),
            'model_name': model_type_name,
            'classification_report': {},
            'confusion_matrix': []
        }

    # Generate dummy AUC, slightly varying for human vs AI
    base_auc = 0.75 if 'human' in model_type_name else 0.78
    mock_test_auc = np.random.uniform(base_auc - 0.05, base_auc + 0.05)
    mock_test_auc = min(0.9, max(0.6, mock_test_auc)) # Keep within reasonable bounds
    
    num_samples = len(y_test)
    mock_predictions_proba = np.random.rand(num_samples) # Random probabilities
    
    # Ensure some spread for ROC curve
    mock_predictions_proba[y_test == 1] = np.random.uniform(0.6, 0.9, size=(y_test == 1).sum())
    mock_predictions_proba[y_test == 0] = np.random.uniform(0.1, 0.4, size=(y_test == 0).sum())


    # Dummy classification report and confusion matrix
    mock_report = {
        "0": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85, "support": int(num_samples * 0.8)},
        "1": {"precision": 0.7, "recall": 0.6, "f1-score": 0.65, "support": int(num_samples * 0.2)},
        "accuracy": 0.82, "macro avg": {}, "weighted avg": {}
    }
    mock_cm = [[int(num_samples * 0.7), int(num_samples * 0.1)],
               [int(num_samples * 0.08), int(num_samples * 0.12)]] # Example CM

    # Dummy feature importance
    # Use features_list to generate dummy importances for actual features
    dummy_importances = np.random.rand(len(features_list))
    dummy_importances = dummy_importances / dummy_importances.sum() # Normalize
    importance_df = pd.DataFrame({
        'feature': features_list,
        'importance': dummy_importances
    }).sort_values('importance', ascending=False).head(10) # Max 10 features for simple mock

    return {
        'auc': mock_test_auc,
        'predictions': mock_predictions_proba,
        'true_labels': y_test.values,
        'feature_importance': importance_df,
        'model_name': model_type_name,
        'classification_report': mock_report,
        'confusion_matrix': mock_cm
    }

# =====================================================================
# MOCK VISUALIZATION AND REPORTING (functions remain mostly same, but use mock data)
# =====================================================================

def plot_roc_curve(y_true, y_preds_human, y_preds_ai, theory, stage, human_auc, ai_auc, output_dir='results'):
    """Plots and saves the ROC curve for human-tuned and AI-tuned models."""
    plt.figure(figsize=(8, 7))
    
    # Ensure y_true has at least two classes to compute ROC curve
    if len(np.unique(y_true)) < 2:
        print(f"  MOCK: Cannot plot ROC curve for {theory} ({stage}): Target has less than 2 classes.")
        plt.close()
        return

    fpr_human, tpr_human, _ = roc_curve(y_true, y_preds_human)
    fpr_ai, tpr_ai, _ = roc_curve(y_true, y_preds_ai)
    
    plt.plot(fpr_human, tpr_human, color='skyblue', lw=2, label=f'Human-Tuned (AUC = {human_auc:.3f})')
    plt.plot(fpr_ai, tpr_ai, color='lightcoral', lw=2, label=f'AI-Tuned (AUC = {ai_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {theory.replace("_", " ").title()} - {stage.capitalize()} Stage (Test Set)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    filepath = os.path.join(output_dir, f'roc_curve_{stage}_{theory}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC curve for {theory} ({stage}) to {filepath}")

def create_comparison_plots(results_dict, stage):
    """Create comprehensive comparison visualizations."""
    print(f"\nCreating aggregated visualization plots for {stage}...")
    
    os.makedirs('results', exist_ok=True)
    
    theories = []
    human_aucs = []
    ai_aucs = []
    improvements = []
    
    for theory, result in results_dict.items():
        theories.append(theory.replace('_', ' ').title())
        human_auc = result['human']['auc']
        ai_auc = result['ai']['auc']
        
        human_aucs.append(human_auc)
        ai_aucs.append(ai_auc)
        improvements.append(ai_auc - human_auc)
    
    if not theories:
        print(f"No valid theory results for {stage} to create comparison plots.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{stage.capitalize()} Stage: Human vs AI-Tuned Random Forest Comparison', 
                 fontsize=16, fontweight='bold')
    
    x = np.arange(len(theories))
    width = 0.35
    
    ax1.bar(x - width/2, human_aucs, width, label='Human-Tuned (Replicated R)', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, ai_aucs, width, label='AI-Tuned', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('Theoretical Models')
    ax1.set_ylabel('AUC Score (Test Set)')
    ax1.set_title('AUC Comparison: Human vs AI-Tuned')
    ax1.set_xticks(x)
    ax1.set_xticklabels(theories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(min(0.5, min(human_aucs + ai_aucs) - 0.05), max(1.0, max(human_aucs + ai_aucs) + 0.05))

    for i, (h, a) in enumerate(zip(human_aucs, ai_aucs)):
        ax1.text(i - width/2, h + 0.005, f'{h:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=8)
    
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(theories, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Theoretical Models')
    ax2.set_ylabel('AUC Improvement (AI - Human)')
    ax2.set_title('AI Improvement Over Human Baseline')
    ax2.set_xticklabels(theories, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.002 if height >= 0 else -0.005),
                f'{imp:+.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    all_aucs_flat = human_aucs + ai_aucs
    labels = ['Human-Tuned (Replicated R)'] * len(human_aucs) + ['AI-Tuned'] * len(ai_aucs)
    df_plot = pd.DataFrame({'AUC': all_aucs_flat, 'Method': labels})
    
    sns.boxplot(data=df_plot, x='Method', y='AUC', ax=ax3)
    sns.swarmplot(data=df_plot, x='Method', y='AUC', ax=ax3, color='black', alpha=0.7)
    ax3.set_title('AUC Distribution Comparison (Test Set)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(min(0.5, min(human_aucs + ai_aucs) - 0.05), max(1.0, max(human_aucs + ai_aucs) + 0.05))

    if ai_aucs: 
        best_theory_idx = np.argmax(ai_aucs)
        best_theory_name = theories[best_theory_idx].replace(' ', '_').lower() 
        
        best_result_dict = results_dict[best_theory_name]
        top_features = best_result_dict['ai']['feature_importance'].head(10)

        if not top_features.empty:
            ax4.barh(range(len(top_features)), top_features['importance'], color='lightgreen', alpha=0.8)
            ax4.set_yticks(range(len(top_features)))
            ax4.set_yticklabels(top_features['feature'], fontsize=8)
            ax4.set_xlabel('Feature Importance')
            ax4.set_title(f'Top Features: {theories[best_theory_idx]} (Best AI Model)')
            ax4.invert_yaxis() 
            ax4.grid(True, alpha=0.3)
        else:
            ax4.set_title(f'No feature importance to plot for {theories[best_theory_idx]}.')
            ax4.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
    else:
        ax4.set_title('No AI models to plot feature importance.')
        ax4.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()
    plot_filepath = os.path.join('results', f'{stage}_comparison_plots.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Aggregated comparison plots saved to: {plot_filepath}")
    
    total_improved = sum(1 for imp in improvements if imp > 0)
    total_theories = len(theories)
    
    positive_improvements_list = [imp for imp in improvements if imp > 0]
    avg_improvement_positive = np.mean(positive_improvements_list) if positive_improvements_list else 0

    overall_avg_improvement = np.mean(improvements) if improvements else 0

    print(f"\n=== {stage.upper()} STAGE SUMMARY ===")
    print(f"Total theoretical models evaluated: {total_theories}")
    print(f"Models where AI beat human: {total_improved}/{total_theories}")
    print(f"Average improvement across all models (AI - Human): {overall_avg_improvement:.4f}")
    if positive_improvements_list:
        print(f"Average improvement for models where AI won: {avg_improvement_positive:.4f}")
    print(f"Best human AUC: {max(human_aucs):.4f}" if human_aucs else "N/A")
    print(f"Best AI AUC: {max(ai_aucs):.4f}" if ai_aucs else "N/A")
    print(f"Maximum individual improvement: {max(improvements):.4f}" if improvements else "N/A")

def generate_detailed_report(all_results):
    """Generate comprehensive analysis report."""
    report_path = 'results/detailed_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE ANALYSIS: LLM vs HUMAN-TUNED RANDOM FORESTS (MOCK RUN)\n")
        f.write("="*80 + "\n\n")
        f.write("Note: This report is generated from a MOCK RUN. All data and metrics are simulated.\n")
        f.write("      'Human-tuned' in this report refers to models replicating R's caret::train defaults\n")
        f.write("      (10-fold CV with downsampling and mtry tuning) on the training set, evaluated on the test set.\n")
        f.write("      'AI-tuned' refers to models with hyperparameters optimized by an LLM on a validation set,\n")
        f.write("      and then the final model is trained on the full training set with optimized parameters,\n")
        f.write("      and evaluated on the same test set.\n\n")
        
        for stage, results in all_results.items():
            f.write(f"\n{'='*30} {stage.upper()} STAGE ANALYSIS {'='*30}\n")
            
            for theory, result in results.items():
                human_auc = result['human']['auc']
                ai_auc = result['ai']['auc']
                improvement = ai_auc - human_auc
                
                f.write(f"\n--- {theory.upper().replace('_', ' ')} Theory ---\n")
                f.write(f"  Features used: {result['features']}\n")
                f.write(f"  Human-tuned AUC (Test Set): {human_auc:.4f}\n")
                f.write(f"  AI-tuned AUC (Test Set):    {ai_auc:.4f}\n")
                f.write(f"  Improvement (AI - Human):   {improvement:+.4f} {'✓' if improvement > 0 else '✗'}\n")
                
                f.write(f"  Best AI Params (used for final AI model): {result['best_ai_params']}\n")

                if 'optimization_history' in result:
                    if len(result['optimization_history']) > 0:
                        successful_iterations = sum(1 for trial in result['optimization_history'] 
                                                  if trial.get('beats_human_baseline_test_auc', False))
                        best_val_auc_llm = max([t['validation_auc'] for t in result['optimization_history']])
                        
                        f.write(f"  LLM Optimization History:\n")
                        f.write(f"    Iterations beating human TEST baseline (Validation AUC): {successful_iterations}/{len(result['optimization_history'])}\n")
                        f.write(f"    Max AI Validation AUC during optimization: {best_val_auc_llm:.4f}\n")
                        
                        f.write(f"    Top 3 LLM Suggested Parameters (by Validation AUC during optimization):\n")
                        sorted_history = sorted(result['optimization_history'], key=lambda x: x['validation_auc'], reverse=True)
                        for i, trial in enumerate(sorted_history[:3]):
                            f.write(f"      {i+1}. Val AUC: {trial['validation_auc']:.4f}, Params: {trial['params']}\n")
                    else:
                        f.write("  No LLM optimization history (skipped or failed).\n")

                f.write(f"\n  Human Model Classification Report (Test Set):\n")
                if result['human']['classification_report']:
                    f.write(json.dumps(result['human']['classification_report'], indent=2) + "\n")
                f.write(f"  Human Model Confusion Matrix (Test Set):\n")
                if result['human']['confusion_matrix']:
                    f.write(str(result['human']['confusion_matrix']) + "\n")

                f.write(f"\n  AI Model Classification Report (Test Set):\n")
                if result['ai']['classification_report']:
                    f.write(json.dumps(result['ai']['classification_report'], indent=2) + "\n")
                f.write(f"  AI Model Confusion Matrix (Test Set):\n")
                if result['ai']['confusion_matrix']:
                    f.write(str(result['ai']['confusion_matrix']) + "\n")

                f.write(f"\n  Top 5 Feature Importances (AI Model):\n")
                if not result['ai']['feature_importance'].empty:
                    for idx, row in result['ai']['feature_importance'].head(5).iterrows():
                        f.write(f"    - {row['feature']}: {row['importance']:.4f}\n")
                else:
                    f.write("    No feature importances available.\n")

                f.write("\n" + "-" * 50 + "\n") # Separator between theories
        f.write(f"\n\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nDetailed report saved to: {report_path}")

def save_data_for_plotting(all_results):
    """Save all relevant data to easily parsable files."""
    print("\nSaving detailed results for independent plotting...")
    output_dir = 'results/raw_data_for_plots'
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []
    
    for stage, results in all_results.items():
        for theory, result_dict in results.items():
            if 'optimization_history' in result_dict and result_dict['optimization_history']:
                history_df = pd.DataFrame(result_dict['optimization_history'])
                history_df['params_str'] = history_df['params'].apply(json.dumps)
                history_df['applied_params_str'] = history_df['applied_params'].apply(json.dumps)
                history_df.drop(columns=['params', 'applied_params'], errors='ignore').to_csv(os.path.join(output_dir, f'ai_optimization_history_{stage}_{theory}.csv'), index=False)
                with open(os.path.join(output_dir, f'ai_optimization_history_{stage}_{theory}.json'), 'w') as f:
                    json.dump(result_dict['optimization_history'], f, indent=4)
            
            if not result_dict['human']['feature_importance'].empty:
                result_dict['human']['feature_importance'].to_csv(os.path.join(output_dir, f'feature_importance_human_{stage}_{theory}.csv'), index=False)
            if not result_dict['ai']['feature_importance'].empty:
                result_dict['ai']['feature_importance'].to_csv(os.path.join(output_dir, f'feature_importance_ai_{stage}_{theory}.csv'), index=False)

            # Ensure predictions and true labels are arrays before creating DataFrame
            preds_human = np.array(result_dict['human']['predictions'])
            true_labels_human = np.array(result_dict['human']['true_labels'])
            if preds_human.size > 0 and true_labels_human.size > 0:
                preds_df_human = pd.DataFrame({
                    'true_labels': true_labels_human,
                    'predictions': preds_human
                })
                preds_df_human.to_csv(os.path.join(output_dir, f'predictions_human_{stage}_{theory}.csv'), index=False)

            preds_ai = np.array(result_dict['ai']['predictions'])
            true_labels_ai = np.array(result_dict['ai']['true_labels'])
            if preds_ai.size > 0 and true_labels_ai.size > 0:
                preds_df_ai = pd.DataFrame({
                    'true_labels': true_labels_ai,
                    'predictions': preds_ai
                })
                preds_df_ai.to_csv(os.path.join(output_dir, f'predictions_ai_{stage}_{theory}.csv'), index=False)

            summary_data.append({
                'stage': stage,
                'theory': theory,
                'human_auc': result_dict['human']['auc'],
                'ai_auc': result_dict['ai']['auc'],
                'improvement': result_dict['ai']['auc'] - result_dict['human']['auc'],
                'best_ai_params': result_dict['best_ai_params']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['best_ai_params_str'] = summary_df['best_ai_params'].apply(json.dumps)
    summary_df.drop(columns=['best_ai_params'], errors='ignore').to_csv(os.path.join(output_dir, 'overall_model_summary.csv'), index=False)
    with open(os.path.join(output_dir, 'overall_model_summary.json'), 'w') as f:
        json.dump(summary_data, f, indent=4)

    print(f"All detailed results saved to {output_dir}")

# =====================================================================
# MAIN EXECUTION (MOCKED)
# =====================================================================

def main_mock():
    parser = argparse.ArgumentParser(description="Compare LLM vs Human-Tuned Random Forest Models (MOCK RUN)")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (ignored in mock)")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM optimization (always true in mock)")
    args = parser.parse_args()
    
    # Force skip_llm for mock run
    args.skip_llm = True
    print("\n--- MOCK RUN IN PROGRESS ---")
    print("All heavy computations (data loading, model training, LLM calls) are simulated.")
    print("This script will generate mock output files and plots for demonstration.")

    client = MockOpenAIClient("dummy_key") # Use mock client
    print("MOCK: OpenAI client initialized successfully (using mock).")
    
    os.makedirs('results', exist_ok=True)
    
    # Clean up old mock results to avoid confusion
    if os.path.exists('results'):
        import shutil
        for item in os.listdir('results'):
            path = os.path.join('results', item)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except Exception as e:
                print(f"Error removing old mock result {path}: {e}")

    df_base = mock_load_and_preprocess_data()
    all_results = {}
    
    for stage in ["onset", "escalation"]:
        print(f"\n{'='*80}")
        print(f"MOCK PROCESSING {stage.upper()} STAGE")
        print(f"{'='*80}")
        
        df_stage = mock_prepare_stage_data(df_base, stage)
        target = STAGE1_TARGET if stage == "onset" else STAGE2_TARGET
        
        if df_stage.empty:
            print(f"MOCK: Skipping {stage} stage due to insufficient data or single target class after preprocessing simulation.")
            continue
        
        train_df_overall, test_df_overall = mock_group_aware_split(df_stage, target, test_size=0.25, random_state=666)
        
        if train_df_overall.empty or test_df_overall.empty:
            print(f"MOCK: Skipping {stage} stage - unable to create valid overall train/test split (simulated).")
            continue

        stage_results = {}
        
        for theory in THEORETICAL_MODELS.keys():
            print(f"\n--- MOCK Processing {theory} model for {stage} stage ---")
            
            features = THEORETICAL_MODELS[theory][stage]
            
            model_features_list = [f for f in features if f in df_stage.columns]
            if not model_features_list:
                print(f"MOCK: No available features for {theory} in stage {stage} in the dataframe. Skipping.")
                continue
            
            # Create dummy X and y for mock evaluation
            X_train_human = pd.DataFrame(np.random.rand(100, len(model_features_list)), columns=model_features_list)
            y_train_human = pd.Series(np.random.choice([0, 1], 100, p=[0.8, 0.2]))
            groups_train_human = pd.Series(np.random.randint(1, 20, 100)) # Dummy groups

            X_test_final = pd.DataFrame(np.random.rand(50, len(model_features_list)), columns=model_features_list)
            y_test_final = pd.Series(np.random.choice([0, 1], 50, p=[0.7, 0.3]))


            if X_train_human.empty or y_train_human.nunique() < 2:
                print(f"MOCK: Insufficient training data for human baseline for {theory} in {stage}. Skipping this theory.")
                continue
            if X_test_final.empty or y_test_final.nunique() < 2:
                print(f"MOCK: Insufficient test data for evaluation for {theory} in {stage}. Skipping this theory.")
                continue
            
            print(f"MOCK: Features for {theory}: {model_features_list}")
            print(f"MOCK: Overall Training samples (simulated): {len(X_train_human)}, Overall Test samples (simulated): {len(X_test_final)}")
            
            human_model = mock_train_human_baseline_replication(X_train_human, y_train_human, groups_train_human, model_features_list)
            human_result = mock_evaluate_model(human_model, X_test_final, y_test_final, f"{theory}_human", model_features_list)
            
            print(f"MOCK: Human-tuned model {theory} ({stage}) Test AUC: {human_result['auc']:.4f}")
            
            theory_result = {
                'human': human_result,
                'features': model_features_list,
                'best_ai_params': HUMAN_RF_FIXED_PARAMS.copy(), 
                'optimization_history': [],
            }
            
            ai_result = human_result # Initialize AI result to human, will update if AI succeeds
            
            if not args.skip_llm: # This will always be false in mock_main
                pass # This block is effectively skipped
            else:
                # Simulate AI optimization process
                # Create dummy X_train_ai_tune, X_val_ai_tune from the mock overall train data
                X_train_ai_tune = pd.DataFrame(np.random.rand(80, len(model_features_list)), columns=model_features_list)
                y_train_ai_tune = pd.Series(np.random.choice([0, 1], 80, p=[0.75, 0.25]))
                X_val_ai_tune = pd.DataFrame(np.random.rand(20, len(model_features_list)), columns=model_features_list)
                y_val_ai_tune = pd.Series(np.random.choice([0, 1], 20, p=[0.6, 0.4]))

                if X_train_ai_tune.empty or y_train_ai_tune.nunique() < 2 or X_val_ai_tune.empty or y_val_ai_tune.nunique() < 2:
                    print(f"MOCK: Insufficient data for AI validation split for {theory} in {stage}. Skipping AI optimization for this theory.")
                    theory_result['ai'] = human_result 
                else:
                    best_ai_params, optimization_history, best_ai_val_auc = mock_llm_hyperparameter_optimization(
                        client, X_train_ai_tune, y_train_ai_tune, X_val_ai_tune, y_val_ai_tune, model_features_list, 
                        theory, stage, human_result['auc']
                    )
                    theory_result['best_ai_params'] = best_ai_params
                    theory_result['optimization_history'] = optimization_history
                    
                    mock_ai_model = MockPipeline([
                        ('preprocessor', mock_create_preprocessor(model_features_list)),
                        ('classifier', MockRandomForestClassifier())
                    ])
                    ai_result = mock_evaluate_model(mock_ai_model, X_test_final, y_test_final, f"{theory}_ai", model_features_list)
                    
                    print(f"MOCK: AI-tuned model {theory} ({stage}) Test AUC: {ai_result['auc']:.4f}")
                    print(f"MOCK: Improvement (AI - Human): {ai_result['auc'] - human_result['auc']:.4f}")
                    
                    theory_result['ai'] = ai_result
            
            stage_results[theory] = theory_result
            
            # Generate individual ROC curve for each theory AFTER final AI-tuned model is developed
            if len(human_result['true_labels']) > 0 and human_result['true_labels'].nunique() > 1: # Check for valid data
                plot_roc_curve(human_result['true_labels'], human_result['predictions'], ai_result['predictions'], 
                               theory, stage, human_result['auc'], ai_result['auc'])

        all_results[stage] = stage_results
        
        if stage_results:
            create_comparison_plots(stage_results, stage)
    
    if all_results:
        save_data_for_plotting(all_results)
        generate_detailed_report(all_results)
        
        print(f"\n{'='*80}")
        print("MOCK FINAL OVERALL SUMMARY")
        print(f"{'='*80}")
        
        for stage, results in all_results.items():
            print(f"\n{stage.upper()} STAGE:")
            improvements = []
            if results: 
                for theory, result in results.items():
                    human_auc = result['human']['auc']
                    ai_auc = result['ai']['auc']
                    improvement = ai_auc - human_auc
                    improvements.append(improvement)
                    
                    status = "✓" if improvement > 0 else "✗"
                    print(f"  {theory:20s}: Human (Test): {human_auc:.4f} → AI (Test): {ai_auc:.4f} ({improvement:+.4f}) {status}")
                
                total_theories_stage = len(improvements)
                positive_improvements_stage = [imp for imp in improvements if imp > 0]
                print(f"  Models improved by AI: {len(positive_improvements_stage)}/{total_theories_stage}")
                
                avg_positive_improvement_stage = np.mean(positive_improvements_stage) if positive_improvements_stage else 0
                avg_overall_improvement_stage = np.mean(improvements) if improvements else 0
                
                print(f"  Avg. Improvement (AI - Human) for this stage: {avg_overall_improvement_stage:.4f}")
                if positive_improvements_stage:
                    print(f"  Avg. Improvement (when AI improved) for this stage: {avg_positive_improvement_stage:.4f}")
                print(f"  Best Human AUC in this stage: {max(human_auc for _, res in results.items() for human_auc in [res['human']['auc']]):.4f}")
                print(f"  Best AI AUC in this stage:    {max(ai_auc for _, res in results.items() for ai_auc in [res['ai']['auc']]):.4f}")
            else:
                print(f"  No models processed for {stage} stage.")
    else:
        print("\nMOCK: No results to summarize. Ensure mock data processing and model training steps completed successfully.")
    
    print(f"\n--- MOCK RUN COMPLETE! ---")
    print(f"All simulated results including plots and raw data are saved in the 'results/' directory.")

if __name__ == "__main__":
    start_time = time.time()
    main_mock()
    end_time = time.time()
    print(f"\nScript completed in {end_time - start_time:.1f} seconds")