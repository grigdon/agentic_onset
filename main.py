import pandas as pd
import numpy as np
import json
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from openai import OpenAI

# Theoretical models and their features as defined in the R script

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

# Predefined human-tuned hyperparameters from R script using n=1000 estimators & the caret package defaults (R)
# If I understand correctly, caret's default tuning uses `mtry` (max_features) tuning which can be replicated using GridSearchCV.
# Class imbalance handling (sampling = "down" in R) 
# Downsampling: a technique to balance class distribution by randomly selecting a subset of the majority class to match the minority class size.

HUMAN_RF_FIXED_PARAMS = {
    'n_estimators': 1000,
    'max_depth': None, 
    'min_samples_split': 2, 
    'min_samples_leaf': 1, 
    'bootstrap': True, 
    'random_state': 666, # Same seed in R script...and yes, I too wish they used a less 'edgy' number.
    'n_jobs': -1,
    'oob_score': False # OOB score is not directly used/reported in caret's CV, and is incompatible with downsampling, so we will have to compromise here.
}

STAGE1_TARGET = "nviol_sdm_onset" # onset
STAGE2_TARGET = "firstescal" # & escalation target

# features with numbers, hence the name NUMERIC_FEATURES
NUMERIC_FEATURES = ['groupsize', 'lsepkin_adjregbase1', 'lnlrgdpcap', 'lnltotpop', 
                   'lv2x_polyarchy', 'numb_rel_grps', 'mounterr', 't_claim', 't_escal']

# features with categories, hence the name CATEGORICAL_FEATURES
CATEGORICAL_FEATURES = ['status_excl', 'lost_autonomy', 'downgr2_aut', 'regaut', 
                       'lgiantoilfield', 'noncontiguous', 'lfederal', 'coldwar']

# you are welcome to change the model to something cheaper. it was approx 75 cents to run this script with gpt-4-turbo. I have not experimented with gpt-3.5-turbo or other lighter models yet.
OPENAI_MODEL = "gpt-4-turbo"
MAX_ITERATIONS = 10

# here is where we get the LLM to write the prompt for the agent
def get_baseline_prompt(theory, stage, baseline_auc, human_params, history, theory_description, best_ai_val_auc):
    stage_description = "early mobilization patterns" if stage == "onset" else "escalation from nonviolent to violent"
    
    # Filter out parameters not relevant for AI (like oob_score which depends on bootstrap)
    display_human_params = {k: v for k, v in human_params.items() if k not in ['oob_score', 'random_state', 'n_jobs']}

    # this is the prompt itself, telling the LLM what to do
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

def load_and_preprocess_data():
    print("\nStarting data load and initial cleanup...")

    # Load the main data file.
    df = pd.read_csv("onset_escalation_data.csv")
    print(f"Dataset loaded. Initial rows/cols: {df.shape}")
    
    # Applying the core filters from the R script: isrelevant and exclacc must be 1 and 0, respectively.
    print("Applying main R script filters (isrelevant=1, exclacc=0)...")
    df_filtered = df[(df['isrelevant'] == 1) & (df['exclacc'] == 0)].copy()
    print(f"Data after initial filtering: {df_filtered.shape}")
    
    return df_filtered

def prepare_stage_data(df, stage):
    print(f"\nPrepping data for the '{stage}' stage...")
    
    target = STAGE1_TARGET if stage == "onset" else STAGE2_TARGET
    
    # Grab all features that *could* be used for this stage across any of our theoretical models.
    all_possible_features = set()
    for theory_features in THEORETICAL_MODELS.values():
        all_possible_features.update(theory_features[stage])
    
    # Include the target variable and the group ID (`gwgroupid`) in our 'must-have-no-NA' list.
    cols_to_check_for_na = list(all_possible_features) + [target, 'gwgroupid']
    
    # Filter this list down to only columns actually present in our dataframe.
    existing_cols_to_check = [col for col in cols_to_check_for_na if col in df.columns]
    
    df_stage = df[existing_cols_to_check].copy()
    
    # This next bit is where we drop rows with any missing values in our key columns.
    # The paper this code is based on does this, so we're sticking to it, even if it means losing some data.
    initial_rows_before_na_drop = df_stage.shape[0]
    df_stage.dropna(subset=existing_cols_to_check, inplace=True)
    print(f"Dropped {initial_rows_before_na_drop - df_stage.shape[0]} rows due to missing values in essential columns for '{stage}'.")

    print(f"Data for '{stage}' stage ready. Shape: {df_stage.shape}")
    print(f"Target distribution for '{stage}': {df_stage[target].value_counts().to_dict()}")
    
    return df_stage

def group_aware_split(df, target_col, test_size=0.25, random_state=666):
    # This split mimics the R script: we make sure entire groups stay together.
    print(f"Splitting data, keeping groups together (test size: {test_size}, seed: {random_state})...")
    
    unique_groups = df['gwgroupid'].unique()
    np.random.seed(random_state)
    
    # How many groups go into the test set?
    num_test_groups = int(len(unique_groups) * test_size)

    # We need to pick some groups for testing.
    test_groups_selected = np.random.choice(unique_groups, size=num_test_groups, replace=False)
    
    # Create the train and test dataframes based on these group IDs.
    train_mask = ~df['gwgroupid'].isin(test_groups_selected) 
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()

    # Just a quick check to make sure our splits aren't totally empty or broken.
    if train_df.empty or test_df.empty or train_df[target_col].nunique() < 2 or test_df[target_col].nunique() < 2:
        print(f"Heads up: This group split resulted in an unusable train/test set (either empty or only one target class).")
        # We'll try a few more times with slightly different seeds if the first one breaks.
        for _attempt in range(3):
            new_seed = random_state + _attempt + 1
            np.random.seed(new_seed)
            test_groups_selected = np.random.choice(unique_groups, size=num_test_groups, replace=False)
            train_mask = ~df['gwgroupid'].isin(test_groups_selected)
            train_df = df[train_mask].copy()
            test_df = df[~train_mask].copy()
            if not train_df.empty and not test_df.empty and train_df[target_col].nunique() >= 2 and test_df[target_col].nunique() >= 2:
                print(f"Success! Found a good split after {_attempt + 1} tries with seed {new_seed}.")
                break
        else:
            # If after a few tries it's still broken, we'll give up for this stage.
            print("Couldn't get a valid group split after multiple attempts. Skipping this part.")
            return pd.DataFrame(), pd.DataFrame()


    print(f"Train groups: {len(train_df['gwgroupid'].unique())} (Samples: {len(train_df)})")
    print(f"Test groups: {len(test_df['gwgroupid'].unique())} (Samples: {len(test_df)})")
    print(f"Train target counts: {train_df[target_col].value_counts().to_dict()}")
    print(f"Test target counts: {test_df[target_col].value_counts().to_dict()}")
    
    return train_df, test_df

def create_preprocessor(features):
    # This sets up how our data gets prepped before hitting the model.
    numeric_features = [f for f in features if f in NUMERIC_FEATURES]
    categorical_features = [f for f in features if f in CATEGORICAL_FEATURES]
    
    transformers = []
    
    if numeric_features:
        # For numbers, we'll fill in any missing bits with the median value.
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median'))
        ])
        transformers.append(('num_pipeline', numeric_transformer, numeric_features))
    
    if categorical_features:
        # For categories, we'll turn them into numbers (one-hot encoding),
        # dropping the first one to avoid issues.
        # We also tell it to ignore categories it hasn't seen before, just in case.
        categorical_transformer = Pipeline([
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        transformers.append(('cat_pipeline', categorical_transformer, categorical_features))
    
    if not transformers:
        # If there are no features to prep, just let the data pass through.
        return 'passthrough'
    
    # This combines all our prepping steps into one neat package.
    return ColumnTransformer(transformers=transformers, remainder='drop')

# =====================================================================
# MODEL TRAINING AND EVALUATION
# =====================================================================

def train_human_baseline_replication(X_train_full, y_train_full, groups_train_full, features):
    # This function trains our "human-tuned" model, trying to match the R script's behavior.
    # It does 10-fold cross-validation, downsamples the training data, and tunes `mtry`.
    print("\n--- Training the human-tuned baseline (matching R) ---")

    preprocessor = create_preprocessor(features)
    
    # We need to downsample the training data to handle class imbalance, just like in R's caret.
    # R does this per CV fold, but here we do it once on the full training set before GridSearchCV.
    print("Downsampling the training dataset for the human model before GridSearchCV...")
    class_counts = y_train_full.value_counts()
    
    # Only downsample if there are at least two classes to balance.
    if y_train_full.nunique() < 2:
        print("Warning: Only one target class in training data. Downsampling skipped.")
        X_train_downsampled = X_train_full
        y_train_downsampled = y_train_full
        groups_train_downsampled = groups_train_full
    else:
        minority_class = y_train_full.value_counts().idxmin()
        majority_class = y_train_full.value_counts().idxmax()
        minority_count = class_counts[minority_class]

        majority_indices = y_train_full[y_train_full == majority_class].index
        minority_indices = y_train_full[y_train_full == minority_class].index

        np.random.seed(HUMAN_RF_FIXED_PARAMS['random_state'])
        downsampled_majority_indices = np.random.choice(
            majority_indices, minority_count, replace=False
        )

        balanced_indices = np.concatenate([downsampled_majority_indices, minority_indices])
        
        X_train_downsampled = X_train_full.loc[balanced_indices].copy()
        y_train_downsampled = y_train_full.loc[balanced_indices].copy()
        groups_train_downsampled = groups_train_full.loc[balanced_indices].copy()

    print(f"Training data after downsampling: {len(y_train_downsampled)} samples (original: {len(y_train_full)}).")
    print(f"Downsampled target counts: {y_train_downsampled.value_counts().to_dict()}")

    human_rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**HUMAN_RF_FIXED_PARAMS))
    ])

    # We only tune `max_features` (mtry) for the human baseline, similar to R's caret.
    param_grid_human = {
        'classifier__max_features': ['sqrt', 'log2', 1.0] # 1.0 means using all features
    }

    # Setting up the GroupShuffleSplit for cross-validation, keeping groups together.
    human_cv = GroupShuffleSplit(n_splits=10, test_size=0.2, random_state=666)

    grid_search_human = GridSearchCV(
        estimator=human_rf_pipeline,
        param_grid=param_grid_human,
        cv=human_cv,
        scoring='roc_auc', # We're optimizing for AUC here
        n_jobs=-1, # Use all your CPU cores for this
        verbose=0,
        refit=True, # Make sure the best model is retrained on all downsampled data
        return_train_score=False
    )
    
    print("Running GridSearchCV for the human baseline. This might take a moment...")
    grid_search_human.fit(X_train_downsampled, y_train_downsampled, groups=groups_train_downsampled)

    print(f"Human baseline: Best parameters found: {grid_search_human.best_params_}")
    print(f"Human baseline: Mean cross-validated AUC: {grid_search_human.best_score_:.4f}")
    
    return grid_search_human.best_estimator_

def llm_hyperparameter_optimization(client, X_train_ai, y_train_ai, X_val_ai, y_val_ai, features, 
                                  theory, stage, human_test_auc):
    # This is where the LLM tries to find better hyperparameters for our model.
    print(f"\n--- Kicking off LLM optimization for {theory} ({stage}) ---")
    
    preprocessor = create_preprocessor(features)
    trial_history = []
    best_ai_val_auc = 0.0 # Keeping track of the best validation AUC found by the AI so far
    best_params_found = HUMAN_RF_FIXED_PARAMS.copy() # Start with human's fixed params as a fallback

    # A friendly description for the LLM about the current theory.
    theory_desc = THEORY_DESCRIPTIONS.get(theory, "An unclassified theoretical model")
    
    # We loop for a set number of iterations, letting the LLM learn and refine its suggestions.
    for iteration in tqdm(range(MAX_ITERATIONS), desc=f"LLM Optimizing {theory} {stage}"):
        # Try to get parameters from the LLM, train, and evaluate.
        prompt = get_baseline_prompt(
            theory=theory,
            stage=stage,
            baseline_auc=human_test_auc, 
            human_params=HUMAN_RF_FIXED_PARAMS, 
            history=json.dumps(trial_history[-3:], indent=2) if trial_history else "No previous attempts.",
            theory_description=theory_desc,
            best_ai_val_auc=best_ai_val_auc 
        )
        
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7 # A bit of creativity, not too random.
        )
        
        content = response.choices[0].message.content
        
        # Pull out the JSON from the LLM's response.
        json_match = re.search(r'\{[^}]*\}', content, re.DOTALL)
        if not json_match:
            print(f"  Iteration {iteration+1}: No valid JSON found in LLM response. Skipping.")
            continue
        
        suggested_params = json.loads(json_match.group(0))
        
        # Make sure the suggested parameters are actually valid for Random Forest.
        if not validate_parameters(suggested_params):
            print(f"  Iteration {iteration+1}: LLM suggested invalid parameters: {suggested_params}. Skipping.")
            continue
        
        # Combine the LLM's suggestions with our fixed parameters.
        current_iter_params = HUMAN_RF_FIXED_PARAMS.copy()
        current_iter_params.update(suggested_params)
        
        # If bootstrap is off, OOB score has to be off too.
        if current_iter_params.get('bootstrap', True) is False:
            current_iter_params['oob_score'] = False
        
        # Build and train the model with these new parameters.
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**current_iter_params))
        ])
        
        model.fit(X_train_ai, y_train_ai)
        val_preds = model.predict_proba(X_val_ai)[:, 1]
        current_val_auc = roc_auc_score(y_val_ai, val_preds)
        
        # Record everything for this iteration.
        trial_record = {
            "iteration": iteration + 1,
            "params": suggested_params, 
            "applied_params": current_iter_params, 
            "validation_auc": current_val_auc,
            "beats_human_baseline_test_auc": current_val_auc > human_test_auc 
        }
        trial_history.append(trial_record)
        
        # If this attempt is better than our current best, update it.
        if current_val_auc > best_ai_val_auc:
            best_ai_val_auc = current_val_auc
            best_params_found = current_iter_params.copy()
        
        print(f"  Iteration {iteration+1}: Val AUC={current_val_auc:.4f} (Best so far: {best_ai_val_auc:.4f}) {'(Improved!)' if current_val_auc > best_ai_val_auc else ''}")
        
    print(f"\nLLM Optimization for {theory} ({stage}) finished.")
    print(f"Best validation AUC from LLM search: {best_ai_val_auc:.4f}")
    print(f"Best parameters LLM found: {best_params_found}")
    return best_params_found, trial_history, best_ai_val_auc

def validate_parameters(params):
    # Just a quick check to make sure the LLM's suggested parameters are valid for Random Forest.
    valid = True
    errors = []

    valid_rf_params = list(RandomForestClassifier().get_params().keys())
    for key in params.keys():
        if key not in valid_rf_params:
            errors.append(f"Bad parameter name: '{key}'")
            valid = False

    # Check types and ranges for specific parameters.
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

def evaluate_model(model, X_test, y_test, model_type_name):
    # Let's see how well our model actually did on the test data.
    print(f"Checking out the {model_type_name} model on the test set...")
    
    # Critical check: if there's no data or only one class, we can't do much.
    if X_test.empty or y_test.empty or y_test.nunique() < 2:
        print(f"Not enough data for {model_type_name} evaluation or only one target class. Skipping detailed metrics.")
        return {
            'auc': 0.5, 
            'predictions': np.array([]),
            'true_labels': np.array([]),
            'feature_importance': pd.DataFrame(),
            'model_name': model_type_name,
            'classification_report': "Not enough data for evaluation.",
            'confusion_matrix': []
        }

    test_preds_proba = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds_proba)
    
    test_preds_binary = (test_preds_proba > 0.5).astype(int) 
    report = classification_report(y_test, test_preds_binary, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, test_preds_binary)
    
    # Grab feature importance to see what variables mattered most.
    classifier = model.named_steps['classifier']
    preprocessor = model.named_steps['preprocessor']
    
    feature_names = []
    # Try to get the real names of the features after preprocessing.
    numeric_features_in_model = []
    categorical_feature_names_out = []

    # This loops through our preprocessor steps to get the actual feature names.
    for name, transformer, features_in_transformer in preprocessor.transformers_:
        if name == 'num_pipeline': # Our numeric pipeline's name
            numeric_features_in_model.extend(features_in_transformer)
        elif name == 'cat_pipeline': # Our categorical pipeline's name
            ohe = transformer.named_steps['onehot']
            categorical_feature_names_out.extend(list(ohe.get_feature_names_out(features_in_transformer)))
    
    feature_names = numeric_features_in_model + categorical_feature_names_out
    
    # Put feature importance into a nice table.
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'auc': test_auc,
        'predictions': test_preds_proba,
        'true_labels': y_test.values,
        'feature_importance': importance_df,
        'model_name': model_type_name,
        'classification_report': report,
        'confusion_matrix': cm.tolist() 
    }

# =====================================================================
# VISUALIZATION AND REPORTING
# =====================================================================

def plot_roc_curve(y_true, y_preds_human, y_preds_ai, theory, stage, human_auc, ai_auc, output_dir='results'):
    # Drawing up the ROC curve to compare models.
    plt.figure(figsize=(8, 7))
    
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
    print(f"  ROC curve for {theory} ({stage}) saved to {filepath}")

def create_comparison_plots(results_dict, stage):
    # Making some nice charts to summarize everything.
    print(f"\nGenerating summary plots for the {stage} stage...")
    
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
        print(f"No results for {stage} to plot. Skipping comparison charts.")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{stage.capitalize()} Stage: Human vs AI-Tuned Random Forest Comparison', 
                 fontsize=16, fontweight='bold')
    
    # Chart 1: Side-by-side AUC
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
    
    # Chart 2: Improvement plot (AI vs Human)
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
    
    # Chart 3: Performance distribution (boxplot)
    all_aucs_flat = human_aucs + ai_aucs
    labels = ['Human-Tuned (Replicated R)'] * len(human_aucs) + ['AI-Tuned'] * len(ai_aucs)
    df_plot = pd.DataFrame({'AUC': all_aucs_flat, 'Method': labels})
    
    sns.boxplot(data=df_plot, x='Method', y='AUC', ax=ax3)
    sns.swarmplot(data=df_plot, x='Method', y='AUC', ax=ax3, color='black', alpha=0.7)
    ax3.set_title('AUC Distribution Comparison (Test Set)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(min(0.5, min(human_aucs + ai_aucs) - 0.05), max(1.0, max(human_aucs + ai_aucs) + 0.05))

    # Chart 4: Top features for the best AI model in this stage.
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
    print(f"Summary plots saved to: {plot_filepath}")
    
    # Final stats for the stage
    total_improved = sum(1 for imp in improvements if imp > 0)
    total_theories = len(theories)
    
    positive_improvements_list = [imp for imp in improvements if imp > 0]
    avg_improvement_positive = np.mean(positive_improvements_list) if positive_improvements_list else 0

    overall_avg_improvement = np.mean(improvements) if improvements else 0

    print(f"\n--- {stage.upper()} STAGE SUMMARY ---")
    print(f"Total models evaluated: {total_theories}")
    print(f"AI models that beat human: {total_improved}/{total_theories}")
    print(f"Average improvement (AI - Human): {overall_avg_improvement:.4f}")
    if positive_improvements_list:
        print(f"Average improvement where AI won: {avg_improvement_positive:.4f}")
    print(f"Best human AUC: {max(human_aucs):.4f}" if human_aucs else "N/A")
    print(f"Best AI AUC: {max(ai_aucs):.4f}" if ai_aucs else "N/A")
    print(f"Max individual improvement: {max(improvements):.4f}" if improvements else "N/A")

# I generated this report template with a prompt to the LLM. Honestly, it did a great job for the time contraints. 
def generate_detailed_report(all_results):
    report_path = 'results/detailed_analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE ANALYSIS: AI vs HUMAN-TUNED RANDOM FORESTS\n")
        f.write("="*80 + "\n\n")
        f.write("Just a note:\n")
        f.write("  'Human-tuned' means models set up like R's caret::train defaults\n")
        f.write("  (10-fold CV with downsampling and mtry tuning) evaluated on the test set.\n")
        f.write("  'AI-tuned' models had their settings optimized by an LLM on a validation set,\n")
        f.write("  then trained on the full training set, and also evaluated on the test set.\n\n")
        
        for stage, results in all_results.items():
            f.write(f"\n{'='*30} {stage.upper()} STAGE BREAKDOWN {'='*30}\n")
            
            for theory, result in results.items():
                human_auc = result['human']['auc']
                ai_auc = result['ai']['auc']
                improvement = ai_auc - human_auc
                
                f.write(f"\n--- {theory.upper().replace('_', ' ')} Theory ---\n")
                f.write(f"  Features used: {result['features']}\n")
                f.write(f"  Human-tuned AUC (Test): {human_auc:.4f}\n")
                f.write(f"  AI-tuned AUC (Test):    {ai_auc:.4f}\n")
                f.write(f"  Improvement (AI - Human):   {improvement:+.4f} {'(AI Win!)' if improvement > 0 else '(Human Win/Draw)'}\n")
                
                f.write(f"  Best AI Params used: {result['best_ai_params']}\n")

                if 'optimization_history' in result and len(result['optimization_history']) > 0:
                    successful_iterations = sum(1 for trial in result['optimization_history'] 
                                              if trial.get('beats_human_baseline_test_auc', False))
                    best_val_auc_llm = max([t['validation_auc'] for t in result['optimization_history']])
                    
                    f.write(f"  AI Optimization Journey:\n")
                    f.write(f"    Times AI's validation AUC beat human's test AUC: {successful_iterations}/{len(result['optimization_history'])}\n")
                    f.write(f"    Highest AI Validation AUC during search: {best_val_auc_llm:.4f}\n")
                    
                    f.write(f"    Top 3 AI Suggested Parameters (by Validation AUC):\n")
                    sorted_history = sorted(result['optimization_history'], key=lambda x: x['validation_auc'], reverse=True)
                    for i, trial in enumerate(sorted_history[:3]):
                        f.write(f"      {i+1}. Val AUC: {trial['validation_auc']:.4f}, Params: {trial['params']}\n")
                else:
                    f.write("  No AI optimization history recorded.\n")

                f.write(f"\n  Human Model Report (Test Set):\n")
                if result['human']['classification_report']:
                    f.write(json.dumps(result['human']['classification_report'], indent=2) + "\n")
                f.write(f"  Human Model Confusion Matrix (Test Set):\n")
                if result['human']['confusion_matrix']:
                    f.write(str(result['human']['confusion_matrix']) + "\n")

                f.write(f"\n  AI Model Report (Test Set):\n")
                if result['ai']['classification_report']:
                    f.write(json.dumps(result['ai']['classification_report'], indent=2) + "\n")
                f.write(f"  AI Model Confusion Matrix (Test Set):\n")
                if result['ai']['confusion_matrix']:
                    f.write(str(result['ai']['confusion_matrix']) + "\n")

                f.write(f"\n  Top 5 Important Features (AI Model):\n")
                if not result['ai']['feature_importance'].empty:
                    for idx, row in result['ai']['feature_importance'].head(5).iterrows():
                        f.write(f"    - {row['feature']}: {row['importance']:.4f}\n")
                else:
                    f.write("    No feature importances available.\n")

                f.write("\n" + "-" * 50 + "\n") 
        f.write(f"\n\nReport finished on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\nFull detailed report saved to: {report_path}")

def save_data_for_plotting(all_results):
    # Dumping all the raw data for anyone who wants to make custom plots.
    print("\nSaving all the detailed data for custom plots...")
    output_dir = 'results/raw_data_for_plots'
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []
    
    for stage, results in all_results.items():
        for theory, result_dict in results.items():
            # Saving AI optimization steps.
            if 'optimization_history' in result_dict and result_dict['optimization_history']:
                history_df = pd.DataFrame(result_dict['optimization_history'])
                history_df['params_str'] = history_df['params'].apply(json.dumps) 
                history_df['applied_params_str'] = history_df['applied_params'].apply(json.dumps) 
                history_df.drop(columns=['params', 'applied_params'], errors='ignore').to_csv(os.path.join(output_dir, f'ai_optimization_history_{stage}_{theory}.csv'), index=False)
                with open(os.path.join(output_dir, f'ai_optimization_history_{stage}_{theory}.json'), 'w') as f:
                    json.dump(result_dict['optimization_history'], f, indent=4)
            
            # Saving how important each feature was.
            if not result_dict['human']['feature_importance'].empty:
                result_dict['human']['feature_importance'].to_csv(os.path.join(output_dir, f'feature_importance_human_{stage}_{theory}.csv'), index=False)
            if not result_dict['ai']['feature_importance'].empty:
                result_dict['ai']['feature_importance'].to_csv(os.path.join(output_dir, f'feature_importance_ai_{stage}_{theory}.csv'), index=False)
            
            # Saving the predictions and actual labels for ROC curves, etc.
            preds_df_human = pd.DataFrame({
                'true_labels': result_dict['human']['true_labels'],
                'predictions': result_dict['human']['predictions']
            })
            preds_df_human.to_csv(os.path.join(output_dir, f'predictions_human_{stage}_{theory}.csv'), index=False)

            preds_df_ai = pd.DataFrame({
                'true_labels': result_dict['ai']['true_labels'],
                'predictions': result_dict['ai']['predictions']
            })
            preds_df_ai.to_csv(os.path.join(output_dir, f'predictions_ai_{stage}_{theory}.csv'), index=False)

            # Gathering summary data for a big overview table.
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
# MAIN FUNCTION DEFINITION -> CALLING MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare LLM vs Human-Tuned Random Forest Models")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM optimization (test with human baselines only)")
    args = parser.parse_args()
    
    # Setting up the OpenAI client connection. I was having issues with the key so I added some error handling. Hope this might help :)
    client = None
    if not args.skip_llm:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("No OpenAI API key found. Skipping AI optimization.")
            args.skip_llm = True
        else:
            try:
                client = OpenAI(api_key=api_key)
                client.models.list() # A quick test to see if the key works.
                print("OpenAI client connected successfully.")
            except Exception as e:
                print(f"Problem connecting to OpenAI: {e}. Skipping AI optimization.")
                args.skip_llm = True
    
    if args.skip_llm:
        print("Skipping AI optimization. Just running human baselines.") # Faster run if we skip the LLM part for testing
    
    os.makedirs('results', exist_ok=True) # Make sure our results folder exists.
    
    # Get the raw data loaded and initially filtered.
    df_base = load_and_preprocess_data()
    all_results = {} # This will hold all our findings.
    
    # Loop through both 'onset' and 'escalation' prediction stages.
    for stage in ["onset", "escalation"]:
        print(f"\n{'='*80}")
        print(f"STARTING {stage.upper()} STAGE ANALYSIS")
        print(f"{'='*80}")
        
        # Prep the data specifically for this stage (handles NAs too).
        df_stage = prepare_stage_data(df_base, stage)
        target = STAGE1_TARGET if stage == "onset" else STAGE2_TARGET
        
        # If we lost too much data in prepping, skip this stage. Does not happen during the run. I was just trying to be safe.
        if df_stage.empty:
            print(f"Not enough data left for {stage} stage after cleaning. Skipping.")
            continue
        
        # Split data into training and test sets, keeping groups together. Funny number again.
        train_df_overall, test_df_overall = group_aware_split(df_stage, target, test_size=0.25, random_state=666)
        
        # If the split failed (not enough data in train/test), skip. Does not happen luckily.
        if train_df_overall.empty or test_df_overall.empty:
            print(f"Couldn't make a valid train/test split for {stage}. Skipping.")
            continue

        stage_results = {}
        
        # Now, let's go through each theoretical model for this stage.
        for theory in THEORETICAL_MODELS.keys():
            print(f"\n--- Working on {theory} model for {stage} stage ---")
            
            # Grab the specific features for this theory and stage.
            features = THEORETICAL_MODELS[theory][stage]
            
            # Make sure all features actually exist in our dataset.
            model_features_list = [f for f in features if f in df_stage.columns]
            if not model_features_list:
                print(f"No usable features found for {theory} in {stage}. Skipping this model.")
                continue
            
            # Prep the final training and test sets.
            X_train_human = train_df_overall[model_features_list].copy()
            y_train_human = train_df_overall[target].copy()
            groups_train_human = train_df_overall['gwgroupid'].copy() 

            X_test_final = test_df_overall[model_features_list].copy()
            y_test_final = test_df_overall[target].copy()

            # Critical checks for model training and evaluation data.
            if X_train_human.empty or y_train_human.nunique() < 2:
                print(f"Not enough training data for human baseline {theory} in {stage}. Skipping.")
                continue
            if X_test_final.empty or y_test_final.nunique() < 2:
                print(f"Not enough test data for evaluation of {theory} in {stage}. Skipping.")
                continue
            
            print(f"Features for {theory}: {model_features_list}")
            print(f"Overall Train samples: {len(X_train_human)}, Overall Test samples: {len(X_test_final)}")
            
            # 1. Train the "human" model.
            human_model = train_human_baseline_replication(X_train_human, y_train_human, groups_train_human, model_features_list)
            human_result = evaluate_model(human_model, X_test_final, y_test_final, f"{theory}_human")
            
            print(f"Human-tuned {theory} ({stage}) Test AUC: {human_result['auc']:.4f}")
            
            # Store human model's results and initial AI params.
            theory_result = {
                'human': human_result,
                'features': model_features_list,
                'best_ai_params': HUMAN_RF_FIXED_PARAMS.copy(), 
                'optimization_history': [],
            }
            
            # 2. Try the AI optimization, if we're not skipping it.
            ai_result = human_result # Default to human's result if AI doesn't run.
            
            if not args.skip_llm:
                # Split our main training data further for AI's own training and validation.
                train_df_for_ai_tune, val_df_for_ai_tune = group_aware_split(train_df_overall, target, test_size=0.2, random_state=777)

                # Critical check for AI's split data.
                if train_df_for_ai_tune.empty or val_df_for_ai_tune.empty or train_df_for_ai_tune[target].nunique() < 2 or val_df_for_ai_tune[target].nunique() < 2:
                    print(f"Not enough data for AI's validation split for {theory} in {stage}. Skipping AI optimization for this one.")
                    theory_result['ai'] = human_result 
                else:
                    X_train_ai_tune = train_df_for_ai_tune[model_features_list].copy()
                    y_train_ai_tune = train_df_for_ai_tune[target].copy()
                    X_val_ai_tune = val_df_for_ai_tune[model_features_list].copy()
                    y_val_ai_tune = val_df_for_ai_tune[target].copy()
                    
                    # Run the LLM's hyperparameter search.
                    best_ai_params, optimization_history, best_ai_val_auc = llm_hyperparameter_optimization(
                        client, X_train_ai_tune, y_train_ai_tune, X_val_ai_tune, y_val_ai_tune, model_features_list, 
                        theory, stage, human_result['auc'] 
                    )
                    theory_result['best_ai_params'] = best_ai_params
                    theory_result['optimization_history'] = optimization_history
                    
                    # Train the *final* AI model on the full training data (same as human model).
                    print(f"\nTraining final AI model for {theory} ({stage}) with its best parameters on the full training set...")
                    preprocessor_ai_final = create_preprocessor(model_features_list)
                    ai_model = Pipeline([
                        ('preprocessor', preprocessor_ai_final),
                        ('classifier', RandomForestClassifier(**best_ai_params))
                    ])
                    
                    ai_model.fit(X_train_human, y_train_human) 
                    ai_result = evaluate_model(ai_model, X_test_final, y_test_final, f"{theory}_ai")
                    
                    print(f"AI-tuned {theory} ({stage}) Test AUC: {ai_result['auc']:.4f}")
                    print(f"Improvement (AI - Human): {ai_result['auc'] - human_result['auc']:+.4f}")
                    
                    theory_result['ai'] = ai_result
                        
            else:
                print(f"AI optimization skipped for {theory} in {stage}. AI results will mirror human baseline.")
                theory_result['ai'] = human_result
            
            stage_results[theory] = theory_result
            
            # Plot the ROC curve for this model comparison.
            plot_roc_curve(y_test_final, human_result['predictions'], ai_result['predictions'], 
                           theory, stage, human_result['auc'], ai_result['auc'])

        all_results[stage] = stage_results
        
        # Generate and save summary plots for this stage.
        if stage_results:
            create_comparison_plots(stage_results, stage)
    
    # Save all the raw data for plotting and generate a detailed report.
    if all_results:
        save_data_for_plotting(all_results)
        generate_detailed_report(all_results)
        
        # A final quick summary.
        print(f"\n{'='*80}")
        print("FINAL SCRIPT SUMMARY")
        print(f"{'='*80}")
        
        # this is a great use case for LLMs in my opinion. generating these types of reports is such a pain, but it saves so much time. there is not much that can go wrong here.
        for stage, results in all_results.items():
            print(f"\n{stage.upper()} STAGE RESULTS:")
            improvements = []
            if results: 
                for theory, result in results.items():
                    human_auc = result['human']['auc']
                    ai_auc = result['ai']['auc']
                    improvement = ai_auc - human_auc
                    improvements.append(improvement)
                    
                    status = "AI improved" if improvement > 0 else "Human better/tied"
                    print(f"  {theory:20s}: Human (Test): {human_auc:.4f} → AI (Test): {ai_auc:.4f} ({improvement:+.4f}) [{status}]")
                
                total_theories_stage = len(improvements)
                positive_improvements_stage = [imp for imp in improvements if imp > 0]
                print(f"  AI improved models: {len(positive_improvements_stage)}/{total_theories_stage}")
                
                avg_positive_improvement_stage = np.mean(positive_improvements_stage) if positive_improvements_stage else 0
                avg_overall_improvement_stage = np.mean(improvements) if improvements else 0
                
                print(f"  Avg. Improvement (AI - Human) for this stage: {avg_overall_improvement_stage:.4f}")
                if positive_improvements_stage:
                    print(f"  Avg. Improvement (when AI won) for this stage: {avg_positive_improvement_stage:.4f}")
                print(f"  Best Human AUC in this stage: {max(human_auc for _, res in results.items() for human_auc in [res['human']['auc']]):.4f}")
                print(f"  Best AI AUC in this stage:    {max(ai_auc for _, res in results.items() for ai_auc in [res['ai']['auc']]):.4f}")
            else:
                print(f"  No models processed for {stage} stage.")
    else:
        print("\nNo results to summarize. Something went wrong during data processing or model training.")
    
    print(f"\nAnalysis done! All reports, plots, and data are in the 'results/' folder.")

# here is our main function call
if __name__ == "__main__":
    start_time = time.time()
    main() 
    end_time = time.time() 
    print(f"\nScript finished in {end_time - start_time:.1f} seconds")