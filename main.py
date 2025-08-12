import argparse
import json
import os
import time
import matplotlib
import pandas as pd
from tqdm import tqdm
from itertools import combinations

# Set Matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

from dotenv import load_dotenv
from openai import OpenAI

# Import all necessary functions
from src.data_processing.loader import load_and_preprocess_data, prepare_stage_data
from src.data_processing.splitter import group_aware_split
from src.data_processing.preprocessor import create_preprocessor
from src.models.random_forest import train_human_baseline_replication, train_randomized_search_baseline, evaluate_model
from src.models.llm_optimizer import llm_hyperparameter_optimization
from src.utils.plotting import plot_roc_curve, create_focused_onset_escalation_plots, create_separate_improvement_plots
from src.utils.reporting import generate_detailed_report, save_data_for_plotting
from src.utils.statistical_tests import perform_delong_comparison, load_prediction_data

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_config(file_path):
    """Loads a JSON configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)

# --- SENSITIVITY ANALYSIS FUNCTION ---
def run_sensitivity_analysis(args, client, configs):
    """
    Runs a prompt sensitivity analysis for the LLM optimizer.
    """
    print("--- STARTING PROMPT SENSITIVITY ANALYSIS ---")

    # Configuration for the sensitivity test
    THEORY_TO_TEST = 'complete'
    PROMPT_VARIATIONS = ['original', 'minimalist', 'prescriptive']
    LLM_ITERATIONS = 5
    OUTPUT_DIR = 'results/sensitivity_analysis'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df_base = load_and_preprocess_data('data/onset_escalation_data.csv')
    all_sensitivity_results = []

    for stage in ["onset", "escalation"]:
        print(f"\n{'='*40}\nTesting Stage: {stage.upper()}\n{'='*40}")
        df_stage, target = prepare_stage_data(df_base, stage, configs['models']['THEORETICAL_MODELS'], 
                                              configs['params']['STAGE1_TARGET'], configs['params']['STAGE2_TARGET'])
        train_df_overall, test_df_overall = group_aware_split(df_stage, target, test_size=0.25, random_state=666)
        train_df_ai, val_df_ai = group_aware_split(train_df_overall, target, test_size=0.2, random_state=777)
        features = configs['models']['THEORETICAL_MODELS'][THEORY_TO_TEST][stage]
        
        for variation in PROMPT_VARIATIONS:
            print(f"\n--- Testing Prompt Variation: '{variation}' for {stage} stage ---")
            
            best_params, _, best_val_auc = llm_hyperparameter_optimization(
                client=client, X_train_ai=train_df_ai[features], y_train_ai=train_df_ai[target],
                X_val_ai=val_df_ai[features], y_val_ai=val_df_ai[target], features=features,
                theory=THEORY_TO_TEST, stage=stage, human_test_auc=0.0,
                fixed_params=configs['params']['HUMAN_RF_FIXED_PARAMS'],
                openai_model=configs['openai']['OPENAI_MODEL'], max_iterations=LLM_ITERATIONS,
                numeric_features=configs['params']['NUMERIC_FEATURES'],
                categorical_features=configs['params']['CATEGORICAL_FEATURES'],
                theory_descriptions=configs['models']['THEORY_DESCRIPTIONS'],
                prompt_variation=variation
            )

            preprocessor = create_preprocessor(features, configs['params']['NUMERIC_FEATURES'], configs['params']['CATEGORICAL_FEATURES'])
            final_model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**best_params))])
            final_model.fit(train_df_overall[features], train_df_overall[target])
            
            eval_result = evaluate_model(final_model, test_df_overall[features], test_df_overall[target], "sensitivity_test")
            final_test_auc = eval_result['auc']
            
            print(f"Result for '{variation}': Best Val AUC={best_val_auc:.4f}, Final Test AUC={final_test_auc:.4f}")

            all_sensitivity_results.append({
                'stage': stage, 'prompt_variation': variation, 'best_validation_auc': best_val_auc,
                'final_test_auc': final_test_auc, 'best_params': json.dumps(best_params)
            })

    results_df = pd.DataFrame(all_sensitivity_results)
    output_path = os.path.join(OUTPUT_DIR, 'prompt_sensitivity_results.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- Prompt Sensitivity Analysis Complete ---")
    print(f"Results saved to '{output_path}'")
    print("\nFinal Results:")
    print(results_df[['stage', 'prompt_variation', 'final_test_auc']].to_string(index=False))

# --- FULL ANALYSIS FUNCTION ---
def run_full_analysis(args, client, configs):
    """
    Runs the full model comparison across all theories and stages.
    """
    print("--- STARTING FULL MODEL COMPARISON RUN ---")
    
    # Unpack configs
    THEORETICAL_MODELS = configs['models']['THEORETICAL_MODELS']
    HUMAN_RF_FIXED_PARAMS = configs['params']['HUMAN_RF_FIXED_PARAMS']
    NUMERIC_FEATURES = configs['params']['NUMERIC_FEATURES']
    CATEGORICAL_FEATURES = configs['params']['CATEGORICAL_FEATURES']
    STAGE1_TARGET = configs['params']['STAGE1_TARGET']
    STAGE2_TARGET = configs['params']['STAGE2_TARGET']
    OPENAI_MODEL = configs['openai']['OPENAI_MODEL']
    MAX_ITERATIONS = configs['openai']['MAX_ITERATIONS']
    THEORY_DESCRIPTIONS = configs['models']['THEORY_DESCRIPTIONS']

    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('results/raw_data_for_plots', exist_ok=True)

    df_base = load_and_preprocess_data('data/onset_escalation_data.csv')
    all_results = {}

    for stage in ["onset", "escalation"]:
        print(f"\n{'='*80}\nSTARTING {stage.upper()} STAGE ANALYSIS\n{'='*80}")
        df_stage, target = prepare_stage_data(df_base, stage, THEORETICAL_MODELS, STAGE1_TARGET, STAGE2_TARGET)
        train_df_overall, test_df_overall = group_aware_split(df_stage, target, test_size=0.25, random_state=666)
        stage_results = {}

        for theory in tqdm(THEORETICAL_MODELS.keys(), desc=f"Processing {stage.capitalize()} Models"):
            features = THEORETICAL_MODELS[theory][stage]
            model_features_list = [f for f in features if f in df_stage.columns]
            if not model_features_list: continue

            X_train_full = train_df_overall[model_features_list].copy()
            y_train_full = train_df_overall[target].copy()
            groups_train_full = train_df_overall['gwgroupid'].copy()
            X_test_final = test_df_overall[model_features_list].copy()
            y_test_final = test_df_overall[target].copy()

            if X_train_full.empty or y_train_full.nunique() < 2 or X_test_final.empty or y_test_final.nunique() < 2: continue

            # 1. Human-Tuned Baseline
            human_model = train_human_baseline_replication(X_train_full, y_train_full, groups_train_full, model_features_list, HUMAN_RF_FIXED_PARAMS, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
            human_result = evaluate_model(human_model, X_test_final, y_test_final, f"{theory}_human")

            # 2. Randomized Search Baseline
            rs_result = None
            if not args.skip_random_search:
                rs_model = train_randomized_search_baseline(X_train_full, y_train_full, groups_train_full, model_features_list, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
                rs_result = evaluate_model(rs_model, X_test_final, y_test_final, f"{theory}_random_search")

            # 3. AI-Tuned (LLM) Model
            ai_result = None
            if not args.skip_llm and client:
                train_df_ai, val_df_ai = group_aware_split(train_df_overall, target, test_size=0.2, random_state=777)
                if not (train_df_ai.empty or val_df_ai.empty or train_df_ai[target].nunique() < 2 or val_df_ai[target].nunique() < 2):
                    best_ai_params, _, _ = llm_hyperparameter_optimization(
                        client, train_df_ai[model_features_list], train_df_ai[target], val_df_ai[model_features_list], val_df_ai[target],
                        model_features_list, theory, stage, human_result['auc'], HUMAN_RF_FIXED_PARAMS, OPENAI_MODEL,
                        MAX_ITERATIONS, NUMERIC_FEATURES, CATEGORICAL_FEATURES, THEORY_DESCRIPTIONS
                    )
                    preprocessor = create_preprocessor(model_features_list, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
                    ai_model = Pipeline([('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**best_ai_params))])
                    ai_model.fit(X_train_full, y_train_full)
                    ai_result = evaluate_model(ai_model, X_test_final, y_test_final, f"{theory}_ai")

            stage_results[theory] = {
                'human': human_result,
                'random_search': rs_result if rs_result else human_result,
                'ai': ai_result if ai_result else human_result
            }

        all_results[stage] = stage_results

    # Post-analysis Reporting and Plotting
    if all_results:
        save_data_for_plotting(all_results)
        generate_detailed_report(all_results)
        # Add calls to plotting functions here if desired
        create_focused_onset_escalation_plots()
        create_separate_improvement_plots()

# --- MAIN DISPATCHER ---
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Conflict Prediction Model Pipeline")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM optimization.")
    parser.add_argument("--skip_random_search", action="store_true", help="Skip Randomized Search baseline.")
    parser.add_argument("--run_sensitivity_analysis", action="store_true", help="Run a prompt sensitivity analysis instead of the full model comparison.")
    args = parser.parse_args()

    configs = {
        'models': load_config('config/theoretical_models.json'),
        'params': load_config('config/default_hyperparameters.json'),
        'openai': load_config('config/openai.json')
    }
    
    client = None
    if not args.skip_llm or args.run_sensitivity_analysis:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
        else:
            print("Warning: OpenAI API key not found. LLM-related tasks will be skipped.")
            args.skip_llm = True

    start_time = time.time()
    
    if args.run_sensitivity_analysis:
        if not client:
            print("Error: Cannot run sensitivity analysis without an OpenAI API key.")
            return
        run_sensitivity_analysis(args, client, configs)
    else:
        run_full_analysis(args, client, configs)

    print(f"\nScript finished in {time.time() - start_time:.1f} seconds.")

if __name__ == "__main__":
    main()