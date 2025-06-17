import pandas as pd
import numpy as np
import json
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import re # added for json/regex parsing

from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from openai import OpenAI

# config

THEORETICAL_MODELS = {
    "grievances": ["lost_autonomy", "downgr2_aut", "status_excl"],
    "political_opportunity": ["regaut", "lv2x_polyarchy", "lfederal"],
    "resource_mobilization": ["lgiantoilfield", "mounterr", "lsepkin_adjregbase1"],
    "pom_gm": [], # This will be populated dynamically
    "pom_rmm": [], # This will be populated dynamically
    "gm_rmm": [], # This will be populated dynamically
    "complete": [], # Will contain all theoretical features
    "base": [] # Will contain only control variables
}

ONSET_CONTROLS = ["coldwar", "groupsize", "lnlrgdpcap", "lnltotpop", "numb_rel_grps", "noncontiguous", "t_claim"]
ESCALATION_CONTROLS = ["coldwar", "groupsize", "lnlrgdpcap", "lnltotpop", "numb_rel_grps", "noncontiguous", "t_escal"]
STAGE1_TARGET = "nviol_sdm_onset"
STAGE2_TARGET = "firstescal"

ALL_NUMERIC_FEATURES = ['groupsize', 'lsepkin_adjregbase1', 'lnlrgdpcap', 'lnltotpop', 'lv2x_polyarchy', 'numb_rel_grps']
ALL_CATEGORICAL_FEATURES = ['status_excl', 'lost_autonomy', 'downgr2_incl', 'downgr2_aut', 'regaut', 'lgiantoilfield', 'mounterr', 'noncontiguous', 'lfederal', 'coldwar']

OPENAI_MODEL = "gpt-4-turbo"
MAX_ITERATIONS = 10

PROMPT_TEMPLATE = """
As an expert in political conflict ML models, optimize Random Forest parameters 
for predicting {stage} using the {theory} theoretical framework.

Current performance:
- Best AUC: {best_auc:.4f}
- Trial history: {history}

Identify ONE new parameter set focusing on:
1. Gaps in the search space
2. Stage-specific dynamics (onset vs escalation)
3. Theoretical feature characteristics

Respond ONLY with JSON using this schema:
{{"n_estimators": int, "max_depth": int|null, 
  "min_samples_split": int, "min_samples_leaf": int}}
"""

REPORT_PROMPT = """
Generate a conflict prediction briefing for political officials using these insights:

Stage: {stage}
Theoretical Model: {theory}
Best AUC: {auc:.4f}
Key Predictors: {top_features}

Stage-Specific Findings:
{stage_insights}

Analysis Recommendations:
1. Focus on {top_features[0]} and {top_features[1]} for policy interventions
2. Consider nonlinear thresholds: {thresholds}
3. Priority regions: {regions}

Format concisely using:
- Key Findings
- Policy Implications
- Monitoring Recommendations
"""

# data loading

def load_and_preprocess_data():
    print("\n=== Loading raw data ===")
    try:
        # Update path to use current directory
        df = pd.read_stata("onset_escalation_data.dta")
    except FileNotFoundError:
        print("Error: The data file 'onset_escalation_data.dta' was not found in the specified path.")
        print("Please update the path in the 'load_and_preprocess_data' function.")
        exit()
        
    print(f"Raw data shape: {df.shape}")
    print("\n=== Applying filters ===")
    filtered = df[(df['isrelevant'] == 1) & (df['exclacc'].fillna(1) == 0) & (df['geo_concentrated'].fillna(0) == 1)].copy()
    print(f"Filtered data shape: {filtered.shape}")
    print("Filtered DataFrame columns:")
    print(filtered.columns.tolist())
    return filtered

def group_aware_split(df, test_size=0.3, random_state=42):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df['gwgroupid']))
    return df.iloc[train_idx], df.iloc[test_idx]

def prepare_stage_data(df, stage):
    print(f"\n=== Preparing {stage} data ===")
    target = STAGE1_TARGET if stage == "onset" else STAGE2_TARGET
    all_base_features = list(set(ALL_NUMERIC_FEATURES + ALL_CATEGORICAL_FEATURES))
    features_to_keep = all_base_features + ['gwgroupid', target]

    # Add the relevant time control for the stage
    time_control = 't_claim' if stage == 'onset' else 't_escal'
    if time_control not in features_to_keep:
        features_to_keep.append(time_control)

    df_stage = df[[f for f in features_to_keep if f in df.columns]].copy()

    # Impute missing values for the stage-specific time control
    if time_control in df_stage.columns:
        df_stage[time_control] = df_stage[time_control].fillna(-1)

    print(f"Shape before NA drop: {df_stage.shape}")
    # Only drop rows with NA in the features to be used (excluding the time control, which is now imputed)
    drop_cols = [col for col in df_stage.columns if col != time_control]
    df_stage = df_stage.dropna(subset=drop_cols)
    print(f"Shape after NA drop: {df_stage.shape}")
    return df_stage

def create_dynamic_preprocessor(numeric_features, categorical_features):
    """Create a preprocessing pipeline for the given dynamic feature lists."""
    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))])
        transformers.append(('num', numeric_transformer, numeric_features))
    
    if categorical_features:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        transformers.append(('cat', categorical_transformer, categorical_features))
        
    if not transformers:
        return "passthrough"
        
    return ColumnTransformer(transformers=transformers, remainder='passthrough')


def llm_guided_hyperparameter_search(client, preprocessor, X_train, y_train, X_val, y_val, stage, theory):
    """LLM-optimized parameter search with a dynamic preprocessor."""
    trial_history = []
    best_auc = 0
    # Initialize with default parameters to ensure a valid model can always be built
    best_params = {'n_estimators': 100, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1}
    
    # Store the initial model's performance to use if LLM fails completely
    initial_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(**best_params, random_state=42, n_jobs=-1))
    ])
    try:
        initial_model.fit(X_train, y_train)
        initial_val_preds = initial_model.predict_proba(X_val)[:, 1]
        best_auc = roc_auc_score(y_val, initial_val_preds)
        print(f"Initial model AUC: {best_auc:.4f}")
    except Exception as e:
        print(f"Initial model fit failed: {e}. Best AUC initialized to 0.")


    for i in tqdm(range(MAX_ITERATIONS), desc=f"LLM Optimization for {theory}"):
        llm_response_content = ""
        try:
            prompt = PROMPT_TEMPLATE.format(
                stage=stage,
                theory=theory,
                best_auc=best_auc,
                history=json.dumps(trial_history[-3:], indent=2)
            )
            
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            llm_response_content = response.choices[0].message.content
            json_match = re.search(r"\{.*\}", llm_response_content, re.DOTALL)
            
            if not json_match:
                print(f"Warning: LLM returned no valid JSON: {llm_response_content}. Skipping.")
                continue
            
            suggested_params = json.loads(json_match.group(0))

            if not validate_parameters(suggested_params):
                print(f"Warning: LLM returned invalid parameters: {suggested_params}. Skipping.")
                continue
                
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(**suggested_params, random_state=42, n_jobs=-1))
            ])
            
            model.fit(X_train, y_train)
            val_preds = model.predict_proba(X_val)[:, 1]
            current_auc = roc_auc_score(y_val, val_preds)
            
            trial_history.append({"params": suggested_params, "auc": current_auc})
            
            if current_auc > best_auc:
                best_auc = current_auc
                best_params = suggested_params
                
        except json.JSONDecodeError as e:
            print(f"LLM iteration {i+1} failed due to JSON decoding error: {e}. Response: {llm_response_content}. Continuing with best known parameters.")
        except Exception as e:
            print(f"LLM iteration {i+1} failed: {e}. Continuing with best known parameters.")
            continue
    
    return best_params, trial_history

def validate_parameters(params):
    """Ensure LLM suggestions are valid."""
    if not isinstance(params, dict): return False
    if params.get('max_depth') is not None and params.get('max_depth', 0) < 1: return False
    if params.get('min_samples_split', 2) < 2: return False
    if params.get('min_samples_leaf', 1) < 1: return False
    if params.get('n_estimators', 0) < 1: return False
    return True

# analysis functions

def partial_dependence_analysis(pipeline_model, X, features_to_plot, theory, stage):
    print(f"Generating partial dependence plots for {theory}...")
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    try:
        # Filter features_to_plot to only include those that are actually in X
        valid_features_to_plot = [f for f in features_to_plot if f in X.columns]

        if not valid_features_to_plot:
            print(f"Warning: No valid numeric features found in X for PDP for {theory}.")
            plt.close(fig)
            return

        PartialDependenceDisplay.from_estimator(
            pipeline_model,         
            X,                      
            features=valid_features_to_plot, 
            kind='both',
            subsample=min(1000, len(X)), 
            n_jobs=-1,
            ax=ax
        )
        plt.suptitle(f"Partial Dependence for {theory.replace('_', ' ').title()}\n({stage.capitalize()} Stage)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(output_dir, f"pdp_{theory}_{stage}.png"))
        print(f"Saved PDP for {theory} in {os.path.join(output_dir, f'pdp_{theory}_{stage}.png')}")
    except ValueError as ve:
        print(f"Could not generate PDP for {features_to_plot} (ValueError): {ve}. This often means features are not numeric or not properly handled.")
    except Exception as e:
        print(f"Could not generate PDP for {features_to_plot}: {e}")
    finally:
        plt.close(fig)

def generate_political_report(client, results, stage, theory, top_features):
    """Generate policy-focused briefing using LLM."""
    insights = ("Higher levels of democracy and economic development are associated with a greater likelihood of nonviolent mobilization.") if stage == "onset" else ("Lower levels of democracy and development, combined with existing grievances, are linked to a higher risk of violent escalation.")
    
    # Ensure top_features has at least two elements for the prompt
    safe_top_features = top_features + ['N/A', 'N/A']

    prompt = REPORT_PROMPT.format(
        stage=stage,
        theory=theory.replace('_', ' ').title(),
        auc=results['test_auc'],
        top_features=safe_top_features[:2],
        stage_insights=insights,
        thresholds=f"Monitor for critical shifts, especially in {safe_top_features[0]}",
        regions="Focus on groups with a documented history of autonomy loss or status downgrades."
    )
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to generate political report: {e}")
        return "LLM report generation failed."

def evaluate_model(model, X_test, y_test, theory, stage):
    test_preds = model.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, test_preds)
    
    classifier = model.named_steps['classifier']
    top_features = []
    
    try:
        preprocessor = model.named_steps['preprocessor']
        if preprocessor != "passthrough" and hasattr(preprocessor, 'get_feature_names_out'):
            feature_names_out = preprocessor.get_feature_names_out()
            if len(classifier.feature_importances_) == len(feature_names_out):
                importances = pd.Series(classifier.feature_importances_, index=feature_names_out)
                top_features = importances.sort_values(ascending=False).index.tolist()[:5]
            else:
                print(f"Warning: Mismatch between feature importances ({len(classifier.feature_importances_)}) and preprocessor output feature names ({len(feature_names_out)}). Falling back to original column names.")
                top_features = list(X_test.columns[:5]) 
        else:
            top_features = list(X_test.columns[:5]) 
    except Exception as e:
        print(f"Error getting feature importances or names: {e}")
        top_features = list(X_test.columns[:5]) 

    current_model_features = list(X_test.columns) 
    numeric_cols_for_pdp = [f for f in ALL_NUMERIC_FEATURES if f in current_model_features]

    if numeric_cols_for_pdp:
        # Pass the full model pipeline to partial_dependence_analysis
        partial_dependence_analysis(model, X_test, numeric_cols_for_pdp, theory, stage)
    else:
        print(f"No numeric features in {theory} model for PDP.")
    
    return {'test_auc': test_auc, 'top_features': top_features}


def main():
    parser = argparse.ArgumentParser(description="Run LLM-Agentic Conflict Prediction Pipeline")
    parser.add_argument("--api_key", type=str, help="OpenAI API key. Can also be set as an environment variable OPENAI_API_KEY.")
    args = parser.parse_args()

    # Prioritize argument, then environment variable.
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key must be provided via --api_key argument or OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)

    all_results = {stage: {} for stage in ["onset", "escalation"]}
    
    df_base = load_and_preprocess_data()
    
    # Correctly define all_interactions feature set
    THEORETICAL_MODELS["pom_gm"] = list(set(
        THEORETICAL_MODELS["grievances"] +
        THEORETICAL_MODELS["political_opportunity"]
    ))

    THEORETICAL_MODELS["pom_rmm"] = list(set(
        THEORETICAL_MODELS["grievances"] +
        THEORETICAL_MODELS["resource_mobilization"]
    ))

    THEORETICAL_MODELS["gm_rmm"] = list(set(
        THEORETICAL_MODELS["resource_mobilization"] +
        THEORETICAL_MODELS["grievances"]
    ))

    # Populate complete model with all theoretical features
    all_theoretical_features = set()
    for theory in ["grievances", "political_opportunity", "resource_mobilization"]:
        all_theoretical_features.update(THEORETICAL_MODELS[theory])
    THEORETICAL_MODELS["complete"] = list(all_theoretical_features)
    
    for stage in ["onset", "escalation"]:
        print(f"\n{'='*60}\nPROCESSING {stage.upper()} STAGE\n{'='*60}")
        
        df_stage = prepare_stage_data(df_base, stage)
        target = STAGE1_TARGET if stage == "onset" else STAGE2_TARGET
        
        if df_stage.empty or df_stage[target].nunique() < 2:
            print(f"Skipping {stage} due to insufficient data after preprocessing.")
            continue
        
        train_df, test_df = group_aware_split(df_stage, random_state=42)
        train_df, val_df = group_aware_split(train_df, random_state=21)

        y_train, y_val, y_test = train_df[target], val_df[target], test_df[target]
        
        print(f"Data splits prepared: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        # Set controls based on stage
        controls = ONSET_CONTROLS if stage == "onset" else ESCALATION_CONTROLS
        THEORETICAL_MODELS["base"] = controls  # Update base model with appropriate controls
        
        for theory in THEORETICAL_MODELS:
            print(f"\n>> Evaluating {theory} model for {stage}")
            
            # For complete model, combine all theoretical features with stage-specific controls
            if theory == "complete":
                # Get all theoretical features
                theoretical_features = THEORETICAL_MODELS["complete"]
                # Get stage-specific controls (excluding the time variable)
                stage_controls = [c for c in controls if c != "t_claim" and c != "t_escal"]
                # Add the appropriate time variable based on stage
                time_var = "t_claim" if stage == "onset" else "t_escal"
                features = list(set(theoretical_features + stage_controls + [time_var]))
            # For base model, use only controls
            elif theory == "base":
                features = controls
            # For other models, combine their features with controls
            else:
                features = list(set(THEORETICAL_MODELS[theory] + controls))
            
            # Dynamically determine feature types for the current theory
            numeric_cols = [f for f in features if f in ALL_NUMERIC_FEATURES]
            categorical_cols = [f for f in features if f in ALL_CATEGORICAL_FEATURES]

            # Select only the features relevant to the current model
            X_train_th, X_val_th, X_test_th = train_df[features], val_df[features], test_df[features]
            
            # Create a preprocessor tailored to the current theory's features
            preprocessor = create_dynamic_preprocessor(numeric_cols, categorical_cols)

            print(f"Starting LLM-guided hyperparameter search for {theory}...")
            llm_params, history = llm_guided_hyperparameter_search(
                client, preprocessor, X_train_th, y_train, X_val_th, y_val, stage, theory
            )
            
            print(f"Final LLM-selected parameters for {theory}: {llm_params}")
            
            llm_model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(**llm_params, random_state=42, n_jobs=-1))
            ])
            
            llm_model.fit(X_train_th, y_train)
            
            llm_results = evaluate_model(llm_model, X_test_th, y_test, theory, stage)
            llm_results['history'] = history
            llm_results['best_params'] = llm_params
            
            print("Generating political report...")
            report = generate_political_report(client, llm_results, stage, theory, llm_results['top_features'])
            
            all_results[stage][theory] = {'llm_tuned': llm_results, 'political_report': report}
            
            output_dir = "reports"
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{theory}_{stage}_report.txt"), "w") as f:
                f.write(f"Theory: {theory}\nStage: {stage}\nBest AUC: {llm_results['test_auc']:.4f}\n\n{report}")
    
    print("\n\n--- LLM-TUNED MODEL PERFORMANCE SUMMARY ---")
    for stage, theories in all_results.items():
        print(f"\n{stage.upper()} STAGE:")
        for theory, results in theories.items():
            if 'llm_tuned' in results and 'test_auc' in results['llm_tuned']:
                print(f"  {theory:>25}: LLM-Tuned AUC = {results['llm_tuned']['test_auc']:.4f}")
            else:
                print(f"  {theory:>25}: No LLM-Tuned AUC (skipped or failed)")

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "llm_tuning_results.json"), "w") as f:
        def default_serializer(o):
            if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        json.dump(all_results, f, indent=4, default=default_serializer)

if __name__ == "__main__":
    start_time = time.time()
    # To run this script:
    # main.py --api_key YOUR_OPENAI_API_KEY
    main()
    print(f"\nTotal execution time: {(time.time() - start_time) / 60:.2f} minutes")
