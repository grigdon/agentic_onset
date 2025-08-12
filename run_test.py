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

# Load environment variables from .env file
from dotenv import load_dotenv 

# Use OPENAI API key from environment variable
from openai import OpenAI

# Import all necessary functions from your src package
from src.data_processing.loader import load_and_preprocess_data, prepare_stage_data
from src.data_processing.splitter import group_aware_split
from src.models.random_forest import train_human_baseline_replication, train_randomized_search_baseline, evaluate_model
from src.models.llm_optimizer import llm_hyperparameter_optimization
from src.utils.plotting import plot_roc_curve
from src.utils.reporting import generate_detailed_report, save_data_for_plotting
from src.utils.statistical_tests import perform_delong_comparison, load_prediction_data

# Import scikit-learn components
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def load_config(file_path):
    """Loads a JSON configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():
    """
    Main function to run a condensed test of the conflict prediction pipeline.
    This script tests all modules with a small subset of models and reduced
    computational intensity, saving results to a separate 'test_run' directory.
    """
    load_dotenv()
    print("--- STARTING CONDENSED TEST RUN ---")

    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Run a condensed test of the model pipeline.")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM optimization test.")
    parser.add_argument("--skip_random_search", action="store_true", help="Skip Randomized Search test.")
    args = parser.parse_args()

    # Define the separate output directory for the test run
    TEST_OUTPUT_DIR = 'results/test_run'
    
    # Define a small subset of theories to test
    THEORIES_TO_TEST = ['base', 'complete']

    # Load necessary configurations
    models_config = load_config('config/theoretical_models.json')
    params_config = load_config('config/default_hyperparameters.json')
    openai_config = load_config('config/openai.json')

    THEORETICAL_MODELS = {k: models_config['THEORETICAL_MODELS'][k] for k in THEORIES_TO_TEST}
    HUMAN_RF_FIXED_PARAMS = params_config['HUMAN_RF_FIXED_PARAMS']
    # Use small iteration counts for the test
    LLM_TEST_ITERATIONS = 1
    RS_TEST_ITERATIONS = 2

    # --- Setup ---
    client = None
    if not args.skip_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            client = OpenAI(api_key=api_key)
            print("OpenAI client connected for test.")
        else:
            print("No OpenAI API key found. Skipping LLM part of the test.")
            args.skip_llm = True

    # Create test output directories
    os.makedirs(f'{TEST_OUTPUT_DIR}/plots', exist_ok=True)
    os.makedirs(f'{TEST_OUTPUT_DIR}/reports', exist_ok=True)
    os.makedirs(f'{TEST_OUTPUT_DIR}/raw_data_for_plots', exist_ok=True)
    
    df_base = load_and_preprocess_data('data/onset_escalation_data.csv')
    all_results = {}

    # --- Main Execution Loop ---
    for stage in ["onset", "escalation"]:
        print(f"\n{'='*40}\nTesting {stage.upper()} STAGE\n{'='*40}")
        df_stage, target = prepare_stage_data(df_base, stage, models_config['THEORETICAL_MODELS'], 
                                              params_config['STAGE1_TARGET'], params_config['STAGE2_TARGET'])
        if df_stage.empty: continue
        train_df_overall, test_df_overall = group_aware_split(df_stage, target, test_size=0.25, random_state=666)
        if train_df_overall.empty: continue
        
        stage_results = {}
        for theory in tqdm(THEORIES_TO_TEST, desc=f"Testing {stage.capitalize()} Models"):
            features = THEORETICAL_MODELS[theory][stage]
            X_train = train_df_overall[features].copy()
            y_train = train_df_overall[target].copy()
            groups_train = train_df_overall['gwgroupid'].copy()
            X_test = test_df_overall[features].copy()
            y_test = test_df_overall[target].copy()

            if X_train.empty: continue
            
            # 1. Human-Tuned Baseline (quick run)
            human_model = train_human_baseline_replication(X_train, y_train, groups_train, features, HUMAN_RF_FIXED_PARAMS,
                                                           params_config['NUMERIC_FEATURES'], params_config['CATEGORICAL_FEATURES'])
            human_result = evaluate_model(human_model, X_test, y_test, f"{theory}_human")

            # 2. Randomized Search Baseline (quick run)
            rs_result = None
            if not args.skip_random_search:
                # Note: The original train_randomized_search_baseline function has a hardcoded n_iter=100.
                # For a true unit/integration test, this would be parameterized. For this test script,
                # we accept that this step will be longer than ideal.
                rs_model = train_randomized_search_baseline(X_train, y_train, groups_train, features,
                                                             params_config['NUMERIC_FEATURES'], params_config['CATEGORICAL_FEATURES'])
                rs_result = evaluate_model(rs_model, X_test, y_test, f"{theory}_random_search")


            # 3. AI-Tuned (LLM) Model (quick run)
            ai_result = None
            if not args.skip_llm:
                train_df_ai, val_df_ai = group_aware_split(train_df_overall, target, test_size=0.2, random_state=777)
                best_params, _, _ = llm_hyperparameter_optimization(
                    client, train_df_ai[features], train_df_ai[target], val_df_ai[features], val_df_ai[target],
                    features, theory, stage, human_result['auc'], HUMAN_RF_FIXED_PARAMS, openai_config['OPENAI_MODEL'],
                    LLM_TEST_ITERATIONS, params_config['NUMERIC_FEATURES'], params_config['CATEGORICAL_FEATURES'], models_config['THEORY_DESCRIPTIONS']
                )
                ai_model = Pipeline([('preprocessor', human_model.named_steps['preprocessor']), ('classifier', RandomForestClassifier(**best_params))])
                ai_model.fit(X_train, y_train)
                ai_result = evaluate_model(ai_model, X_test, y_test, f"{theory}_ai")

            # Store results
            stage_results[theory] = {
                'human': human_result,
                'random_search': rs_result if rs_result else human_result,
                'ai': ai_result if ai_result else human_result,
                'features': features
            }

        all_results[stage] = stage_results

    # --- Reporting and Plotting ---
    if all_results:
        print("\nTest models trained. Now testing reporting and plotting functions...")
        
        # Use the test output directory for all generated files
        test_raw_data_dir = f'{TEST_OUTPUT_DIR}/raw_data_for_plots'
        test_reports_dir = f'{TEST_OUTPUT_DIR}/reports'
        test_plots_dir = f'{TEST_OUTPUT_DIR}/plots'
        
        save_data_for_plotting(all_results, output_dir=test_raw_data_dir)
        generate_detailed_report(all_results, output_dir=test_reports_dir)
        
        print(f"Test completed successfully. Results are in '{TEST_OUTPUT_DIR}'")
    else:
        print("Test failed: No results were generated.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nTest script finished in {end_time - start_time:.1f} seconds.")