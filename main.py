import argparse
import json
import os
import time
import matplotlib
import pandas as pd

# Set Matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use('Agg')

# load environment variables from .env file
from dotenv import load_dotenv 

# use OPENAI API key from environment variable
from openai import OpenAI

# Import modularized functions from src sub-packages
from src.data_processing.loader import load_and_preprocess_data, prepare_stage_data
from src.data_processing.splitter import group_aware_split
from src.data_processing.preprocessor import create_preprocessor
from src.models.random_forest import train_human_baseline_replication, evaluate_model
from src.models.llm_optimizer import llm_hyperparameter_optimization
from src.utils.plotting import (
    plot_roc_curve, create_comparison_plots, create_hyperparameter_optimization_plots,
    create_detailed_parameter_evolution_plots, create_onset_escalation_plots,
    create_focused_onset_escalation_plots, create_variable_importance_plots,
    create_separate_auc_plots, create_separate_improvement_plots # Added all new plot imports
)
from src.utils.reporting import (
    generate_detailed_report, save_data_for_plotting, create_stage_theory_policy_reports,
    create_comparison_table, save_detailed_delong_results, create_onesided_text_report,
    create_pairwise_all_models_report, load_optimization_history, load_results_data, # Added DeLong reporting and history/results loader
)
from src.utils.statistical_tests import load_prediction_data, perform_delong_comparison

# Added imports for Pipeline and RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np # Ensure numpy is imported for array_equal


def load_config(file_path):
    """Loads a JSON configuration file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def main():

    load_dotenv()

    """
    Main function to run the conflict prediction model analysis.
    Orchestrates data loading, model training (human and AI-tuned),
    evaluation, and report/plot generation.
    """
    parser = argparse.ArgumentParser(description="Compare LLM vs Human-Tuned Random Forest Models")
    parser.add_argument("--api_key", type=str, help="OpenAI API key")
    parser.add_argument("--skip_llm", action="store_true", help="Skip LLM optimization (test with human baselines only)")
    args = parser.parse_args()

    # Load configurations from JSON files
    models_config = load_config('config/theoretical_models.json')
    params_config = load_config('config/default_hyperparameters.json')
    openai_config = load_config('config/openai.json')

    THEORETICAL_MODELS = models_config['THEORETICAL_MODELS']
    THEORY_DESCRIPTIONS = models_config['THEORY_DESCRIPTIONS']
    HUMAN_RF_FIXED_PARAMS = params_config['HUMAN_RF_FIXED_PARAMS']
    STAGE1_TARGET = params_config['STAGE1_TARGET']
    STAGE2_TARGET = params_config['STAGE2_TARGET']
    NUMERIC_FEATURES = params_config['NUMERIC_FEATURES']
    CATEGORICAL_FEATURES = params_config['CATEGORICAL_FEATURES']
    OPENAI_MODEL = openai_config['OPENAI_MODEL']
    MAX_ITERATIONS = openai_config['MAX_ITERATIONS']

    # Initialize OpenAI client
    client = None
    if not args.skip_llm:
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("No OpenAI API key found. Skipping AI optimization.")
            args.skip_llm = True
        else:
            try:
                client = OpenAI(api_key=api_key)
                client.models.list() # Test API key validity
                print("OpenAI client connected successfully.")
            except Exception as e:
                print(f"Problem connecting to OpenAI: {e}. Skipping AI optimization.")
                args.skip_llm = True

    # Ensure results directories exist
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/reports', exist_ok=True)
    os.makedirs('results/raw_data_for_plots', exist_ok=True)

    # Load and preprocess initial data
    df_base = load_and_preprocess_data('data/onset_escalation_data.csv')
    all_results = {} # Stores results for all stages and theories

    for stage in ["onset", "escalation"]:
        print(f"\n{'='*80}")
        print(f"STARTING {stage.upper()} STAGE ANALYSIS")
        print(f"{'='*80}")

        df_stage, target = prepare_stage_data(df_base, stage, THEORETICAL_MODELS, STAGE1_TARGET, STAGE2_TARGET)

        if df_stage.empty:
            print(f"Not enough data left for {stage} stage after cleaning. Skipping.")
            continue

        train_df_overall, test_df_overall = group_aware_split(df_stage, target, test_size=0.25, random_state=666)

        if train_df_overall.empty or test_df_overall.empty:
            print(f"Couldn't make a valid train/test split for {stage}. Skipping.")
            continue

        stage_results = {}

        for theory in THEORETICAL_MODELS.keys():
            print(f"\n--- Working on {theory} model for {stage} stage ---")

            features = THEORETICAL_MODELS[theory][stage]
            model_features_list = [f for f in features if f in df_stage.columns]
            if not model_features_list:
                print(f"No usable features found for {theory} in {stage}. Skipping this model.")
                continue

            X_train_human = train_df_overall[model_features_list].copy()
            y_train_human = train_df_overall[target].copy()
            groups_train_human = train_df_overall['gwgroupid'].copy()

            X_test_final = test_df_overall[model_features_list].copy()
            y_test_final = test_df_overall[target].copy()

            if X_train_human.empty or y_train_human.nunique() < 2:
                print(f"Not enough training data for human baseline {theory} in {stage}. Skipping.")
                continue
            if X_test_final.empty or y_test_final.nunique() < 2:
                print(f"Not enough test data for evaluation of {theory} in {stage}. Skipping.")
                continue

            print(f"Features for {theory}: {model_features_list}")
            print(f"Overall Train samples: {len(X_train_human)}, Overall Test samples: {len(X_test_final)}")

            # Train the human-tuned model
            human_model = train_human_baseline_replication(X_train_human, y_train_human, groups_train_human,
                                                           model_features_list, HUMAN_RF_FIXED_PARAMS,
                                                           NUMERIC_FEATURES, CATEGORICAL_FEATURES)
            human_result = evaluate_model(human_model, X_test_final, y_test_final, f"{theory}_human")

            print(f"Human-tuned {theory} ({stage}) Test AUC: {human_result['auc']:.4f}")

            theory_result = {
                'human': human_result,
                'features': model_features_list,
                'best_ai_params': HUMAN_RF_FIXED_PARAMS.copy(), # Default to human params
                'optimization_history': [],
            }

            ai_result = human_result # Default to human's result if AI optimization is skipped

            # Run AI optimization if not skipped
            if not args.skip_llm:
                train_df_for_ai_tune, val_df_for_ai_tune = group_aware_split(train_df_overall, target, test_size=0.2, random_state=777)

                if train_df_for_ai_tune.empty or val_df_for_ai_tune.empty or train_df_for_ai_tune[target].nunique() < 2 or val_df_for_ai_tune[target].nunique() < 2:
                    print(f"Not enough data for AI's validation split for {theory} in {stage}. Skipping AI optimization for this one.")
                    theory_result['ai'] = human_result
                else:
                    X_train_ai_tune = train_df_for_ai_tune[model_features_list].copy()
                    y_train_ai_tune = train_df_for_ai_tune[target].copy()
                    X_val_ai_tune = val_df_for_ai_tune[model_features_list].copy()
                    y_val_ai_tune = val_df_for_ai_tune[target].copy()

                    best_ai_params, optimization_history, best_ai_val_auc = llm_hyperparameter_optimization(
                        client, X_train_ai_tune, y_train_ai_tune, X_val_ai_tune, y_val_ai_tune, model_features_list,
                        theory, stage, human_result['auc'], HUMAN_RF_FIXED_PARAMS, OPENAI_MODEL, MAX_ITERATIONS,
                        NUMERIC_FEATURES, CATEGORICAL_FEATURES, THEORY_DESCRIPTIONS
                    )
                    theory_result['best_ai_params'] = best_ai_params
                    theory_result['optimization_history'] = optimization_history

                    # Train the final AI model on the full training data
                    print(f"\nTraining final AI model for {theory} ({stage}) with its best parameters on the full training set...")
                    preprocessor_ai_final = create_preprocessor(model_features_list, NUMERIC_FEATURES, CATEGORICAL_FEATURES)
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

            # Plot ROC curve for this model comparison
            plot_roc_curve(y_test_final, human_result['predictions'], ai_result['predictions'],
                           theory, stage, human_result['auc'], ai_result['auc'])

        all_results[stage] = stage_results

        # Generate and save summary plots for this stage
        if stage_results:
            create_comparison_plots(stage_results, stage)

    # --- Post-analysis Reporting and Plotting ---
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_dir_path = 'results/raw_data_for_plots' # Path to where prediction CSVs are saved

    if all_results:
        save_data_for_plotting(all_results)
        generate_detailed_report(all_results)

        # Generate LLM-driven policy briefs
        if client is not None:
            create_stage_theory_policy_reports(client, all_results, OPENAI_MODEL, THEORY_DESCRIPTIONS)

        # Generate Hyperparameter Optimization Plots
        print("\nGenerating hyperparameter optimization plots...")
        create_hyperparameter_optimization_plots(list(THEORETICAL_MODELS.keys()), ["onset", "escalation"], load_history_func=load_optimization_history)
        create_detailed_parameter_evolution_plots(["base", "complete"], ["onset", "escalation"], load_history_func=load_optimization_history)
        
        # Generate general comparison plots
        print("\nGenerating general comparison plots...")
        create_onset_escalation_plots()
        create_focused_onset_escalation_plots()
        create_variable_importance_plots()
        create_separate_auc_plots()
        create_separate_improvement_plots()


        # Perform DeLong Tests and generate reports
        print("\nPerforming DeLong tests and generating reports...")
        all_delong_results = {}
        pairwise_delong_results = [] # To store results for pairwise comparisons

        # First, perform AI vs Human DeLong tests
        for stage in ["onset", "escalation"]:
            all_delong_results[stage] = {}
            for theory in THEORETICAL_MODELS.keys():
                ai_preds_file = os.path.join(results_dir_path, f"predictions_ai_{stage}_{theory}.csv")
                human_preds_file = os.path.join(results_dir_path, f"predictions_human_{stage}_{theory}.csv")

                if os.path.exists(ai_preds_file) and os.path.exists(human_preds_file):
                    y_true_ai, y_pred_ai = load_prediction_data(ai_preds_file)
                    y_true_human, y_pred_human = load_prediction_data(human_preds_file)

                    if y_true_ai is not None and y_true_human is not None and np.array_equal(y_true_ai, y_true_human):
                        delong_result = perform_delong_comparison(
                            y_true_ai, y_pred_ai, y_pred_human,
                            f"AI_{theory}", f"Human_{theory}",
                            debug_label='base' if theory == 'base' else None
                        )
                        if delong_result:
                            all_delong_results[stage][theory] = delong_result
                            print(f"✓ DeLong test for {stage} - {theory} (AI vs Human) completed.")
                    else:
                        print(f"Skipping DeLong test for {stage} - {theory} due to missing data or mismatched true labels.")
                else:
                    print(f"Skipping DeLong test for {stage} - {theory} as prediction files not found.")

        # Then, perform pairwise AI vs AI DeLong tests
        for stage in ["onset", "escalation"]:
            stage_ai_models = []
            for theory in THEORETICAL_MODELS.keys():
                ai_preds_file = os.path.join(results_dir_path, f"predictions_ai_{stage}_{theory}.csv")
                if os.path.exists(ai_preds_file):
                    y_true, y_pred = load_prediction_data(ai_preds_file)
                    if y_true is not None:
                        stage_ai_models.append({'theory': theory, 'y_true': y_true, 'y_pred': y_pred})

            if len(stage_ai_models) >= 2:
                for i in range(len(stage_ai_models)):
                    for j in range(i + 1, len(stage_ai_models)):
                        model1 = stage_ai_models[i]
                        model2 = stage_ai_models[j]

                        # Ensure true labels are identical for pairwise comparison
                        if np.array_equal(model1['y_true'], model2['y_true']):
                            pairwise_delong_result = perform_delong_comparison(
                                model1['y_true'], model1['y_pred'], model2['y_pred'],
                                f"AI_{model1['theory']}", f"AI_{model2['theory']}"
                            )
                            if pairwise_delong_result:
                                pairwise_delong_result['stage'] = stage # Add stage info
                                pairwise_delong_results.append(pairwise_delong_result)
                                print(f"✓ Pairwise DeLong test for {stage}: AI_{model1['theory']} vs AI_{model2['theory']} completed.")
                        else:
                            print(f"Skipping pairwise DeLong test for {stage}: AI_{model1['theory']} vs AI_{model2['theory']} due to mismatched true labels.")
            else:
                print(f"Not enough AI models for pairwise comparison in {stage} stage.")


        # Generate DeLong test reports
        if all_delong_results:
            for stage in ["onset", "escalation"]:
                if stage in all_delong_results and all_delong_results[stage]:
                    results_list = list(all_delong_results[stage].values())
                    table_df = create_comparison_table(results_list, stage)
                    if table_df is not None:
                        table_path = os.path.join('results/reports', f"delong_test_proc_style_{stage}_{timestamp}.csv")
                        table_df.to_csv(table_path, index=False)
                        print(f"Table saved to: {table_path}")

            save_detailed_delong_results(all_delong_results, timestamp)
            create_onesided_text_report(all_delong_results, timestamp)

        if pairwise_delong_results:
            create_pairwise_all_models_report(pairwise_delong_results, timestamp)


        # Final summary output to console
        print(f"\n{'='*80}")
        print("FINAL SCRIPT SUMMARY")
        print(f"{'='*80}")

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

                avg_positive_improvement_stage = sum(positive_improvements_stage) / len(positive_improvements_stage) if positive_improvements_stage else 0
                avg_overall_improvement_stage = sum(improvements) / len(improvements) if improvements else 0

                print(f"  Avg. Improvement (AI - Human) for this stage: {avg_overall_improvement_stage:.4f}")
                if positive_improvements_stage:
                    print(f"  Avg. Improvement (when AI won) for this stage: {avg_positive_improvement_stage:.4f}")
                print(f"  Best Human AUC in this stage: {max(human_auc for _, res in results.items() for human_auc in [res['human']['auc']]):.4f}")
                print(f"  Best AI AUC in this stage:    {max(ai_auc for _, res in results.items() for ai_auc in [res['ai']['auc']]):.4f}")
            else:
                print(f"  No models processed for {stage} stage.")
    else:
        print("\nNo results to summarize. Something went wrong during data processing or model training.")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"\nScript finished in {end_time - start_time:.1f} seconds")
