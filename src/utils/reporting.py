import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from pathlib import Path

def generate_detailed_report(all_results, output_dir='results/reports'):
    """
    Generates a comprehensive text report summarizing the performance of
    human-tuned and AI-tuned Random Forest models across all stages and theories.

    Args:
        all_results (dict): A nested dictionary containing all evaluation results.
        output_dir (str): Directory to save the detailed report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'detailed_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE ANALYSIS: AI vs HUMAN-TUNED RANDOM FORESTS\n")
        f.write("="*80 + "\n\n")
        f.write("Just a note:\n")
        f.write("  'Human-tuned' means models set up like R's caret::train defaults\n")
        f.write("  (10-fold CV with downsampling and mtry tuning) evaluated on the test set.\n")
        f.write("  'AI-tuned' models had their settings optimized by an LLM on a validation set,\n")
        f"  then trained on the full training set, and also evaluated on the test set.\n\n"

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

def save_data_for_plotting(all_results, output_dir='results/raw_data_for_plots'):
    """
    Saves raw data, feature importances, and predictions for plotting and detailed analysis.

    Args:
        all_results (dict): A nested dictionary containing all evaluation results.
        output_dir (str): Directory to save the raw data files.
    """
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

def generate_policy_brief_prompt(stage, theory, result, theory_description):
    """
    Generates a prompt for the LLM to create a policy brief based on model insights.

    Args:
        stage (str): The prediction stage ('onset' or 'escalation').
        theory (str): The name of the theoretical model.
        result (dict): The evaluation results for the specific model.
        theory_description (str): Description of the theoretical model.

    Returns:
        str: The formatted prompt for the LLM.
    """
    top_features = result['ai']['feature_importance'].head(5)['feature'].tolist()
    best_auc = max(result['ai']['auc'], result['human']['auc'])

    # These are hardcoded in the original script provided by the user.
    # In a fully modularized system, these might come from a config file
    # or be dynamically generated. For now, keeping as provided.
    stage_insights = {
        "onset": "Early mobilization patterns show that economic and demographic factors are critical predictors. Groups with larger populations and higher GDP per capita are more likely to mobilize, while mountainous terrain provides strategic advantages.",
        "escalation": "Escalation from nonviolent to violent conflict is strongly influenced by political factors and group grievances. Status exclusion and autonomy loss are key triggers, while political system characteristics can either facilitate or prevent escalation."
    }

    thresholds = "Economic development levels and group size thresholds vary by region"
    regions = "Focus on regions with high ethnic diversity and recent autonomy changes"

    REPORT_PROMPT = f"""
Generate a conflict prediction briefing for political officials using these insights:

Stage: {stage}
Theoretical Model: {theory.replace('_', ' ').title()}
Best AUC: {best_auc:.4f}
Key Predictors: {top_features}

Stage-Specific Findings:
{stage_insights[stage]}

Analysis Recommendations:
1. Focus on {top_features[0]} and {top_features[1]} for policy interventions
2. Consider nonlinear thresholds: {thresholds}
3. Priority regions: {regions}

Format concisely using:
- Key Findings
- Policy Implications
- Monitoring Recommendations

Additional Context:
- Human-tuned AUC: {result['human']['auc']:.4f}
- AI-tuned AUC: {result['ai']['auc']:.4f}
- Improvement: {result['ai']['auc'] - result['human']['auc']:+.4f}
- Theory Description: {theory_description}
- Best AI Parameters: {result['best_ai_params']}

Top 5 AI Model Features (by importance):
{result['ai']['feature_importance'].head(5).to_string(index=False)}

IMPORTANT: Your response must follow this EXACT format and include ALL the information above. Start with the Stage, Theoretical Model, Best AUC, and Key Predictors exactly as shown. Then include the Stage-Specific Findings, Analysis Recommendations, and format your response with Key Findings, Policy Implications, and Monitoring Recommendations sections.
"""

    return REPORT_PROMPT

def create_stage_theory_policy_reports(client, all_results, openai_model, theory_descriptions, output_dir='results/reports/policy_briefs'):
    """
    Generates and saves policy briefs for each stage and theory using the LLM.

    Args:
        client (openai.OpenAI): OpenAI API client instance.
        all_results (dict): A nested dictionary containing all evaluation results.
        openai_model (str): Name of the OpenAI model to use for brief generation.
        theory_descriptions (dict): Dictionary of theoretical model descriptions.
        output_dir (str): Directory to save the policy briefs.
    """
    os.makedirs(output_dir, exist_ok=True)
    for stage, results in all_results.items():
        for theory, result in results.items():
            theory_description = theory_descriptions.get(theory, "No description available.")
            prompt = generate_policy_brief_prompt(stage, theory, result, theory_description)
            try:
                response = client.chat.completions.create(
                    model=openai_model, # Use passed openai_model from config
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                content = response.choices[0].message.content
            except Exception as e:
                content = f"[Error generating policy brief: {e}]"
            filename = os.path.join(output_dir, f"policy_brief_{stage}_{theory}.txt")
            with open(filename, 'w') as f:
                f.write(content)

# --- DeLong Test Reporting Functions (Moved from original delong_test_proc_style.py) ---

def create_comparison_table(results, stage, output_dir='results/reports'):
    """
    Creates a comparison table of DeLong test results for AI vs Human models.

    Args:
        results (list): List of dictionaries, each containing results from perform_delong_comparison.
        stage (str): The current prediction stage ('onset' or 'escalation').
        output_dir (str): Directory to save the table CSV.

    Returns:
        pd.DataFrame or None: DataFrame of the comparison table, or None if no results.
    """
    if not results:
        return None

    table_data = []
    for result in results:
        if result is None:
            continue

        table_data.append({
            'AI Model': result['model1_name'],
            'Human Model': result['model2_name'],
            'AI AUC': f"{result['auc1']:.3f}",
            'Human AUC': f"{result['auc2']:.3f}",
            'AUC Difference (AI-Human)': f"{result['auc_diff']:+.3f}",
            '95% CI Lower': f"{result['ci_lower']:.3f}",
            '95% CI Upper': f"{result['ci_upper']:.3f}",
            'Delong Statistic': f"{result['delong_statistic']:.3f}" if result['delong_statistic'] is not None else 'N/A',
            'P-value (One-sided)': f"{result['p_value_one_sided']:.4f}" if result['p_value_one_sided'] is not None else 'N/A'
        })

    if table_data:
        df = pd.DataFrame(table_data)
        # No print statement here, main.py will handle printing
        return df
    return None

def save_detailed_delong_results(all_delong_results, timestamp, output_dir='results/reports'):
    """
    Saves detailed DeLong test results to a comprehensive CSV file.

    Args:
        all_delong_results (dict): Dictionary containing all DeLong test results.
        timestamp (str): Timestamp for naming the file.
        output_dir (str): Directory to save the CSV.

    Returns:
        pd.DataFrame or None: DataFrame of detailed results, or None if no results.
    """
    detailed_data = []
    os.makedirs(output_dir, exist_ok=True)

    for stage in ['onset', 'escalation']:
        if stage not in all_delong_results:
            continue

        for model_type, result in all_delong_results[stage].items():
            if result is None:
                continue

            detailed_data.append({
                'Stage': stage,
                'Model_Type': model_type,
                'Model1_Name': result['model1_name'],
                'Model2_Name': result['model2_name'],
                'AUC1': result['auc1'],
                'AUC2': result['auc2'],
                'AUC_Difference': result['auc_diff'],
                'CI_Lower': result['ci_lower'],
                'CI_Upper': result['ci_upper'],
                'Delong_Statistic': result['delong_statistic'],
                'P_value_Two_sided': result['p_value_two_sided'],
                'P_value_One_sided': result['p_value_one_sided'],
                'Significant_Model1_Better': result['significant']
            })

    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = os.path.join(output_dir, f"detailed_delong_test_results_{timestamp}.csv")
        detailed_df.to_csv(detailed_path, index=False)
        return detailed_df
    return None

def create_onesided_text_report(all_delong_results, timestamp, output_dir='results/reports'):
    """
    Creates a text report summarizing one-sided DeLong test results (Model1 > Model2).

    Args:
        all_delong_results (dict): Dictionary containing all DeLong test results.
        timestamp (str): Timestamp for naming the file.
        output_dir (str): Directory to save the report.

    Returns:
        str: Path to the generated report.
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ONE-SIDED DELONG TEST RESULTS")
    report_lines.append("Testing: Model1 > Model2 (alternative='greater')")
    report_lines.append("=" * 80)
    report_lines.append("")

    os.makedirs(output_dir, exist_ok=True)

    for stage in ['onset', 'escalation']:
        if stage not in all_delong_results:
            continue

        report_lines.append(f"{stage.upper()} STAGE")
        report_lines.append("-" * 40)

        stage_results = []
        for model_type, result in all_delong_results[stage].items():
            if result is None:
                continue

            stage_results.append({
                'model_type': model_type,
                'model1_name': result['model1_name'],
                'model2_name': result['model2_name'],
                'auc1': result['auc1'],
                'auc2': result['auc2'],
                'auc_diff': result['auc_diff'],
                'delong_stat': result['delong_statistic'],
                'p_value': result['p_value_one_sided'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'significant': result['significant']
            })

        # Sort by p-value (most significant first)
        stage_results.sort(key=lambda x: x['p_value'] if x['p_value'] is not None else float('inf'))

        for result in stage_results:
            significance = "***" if result['significant'] else ""
            report_lines.append(f"{result['model1_name']:25} | AUC1: {result['auc1']:.3f} | AUC2: {result['auc2']:.3f} | Z: {result['delong_stat']:6.2f} | p: {result['p_value']:.4f} | CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] {significance}")

        # Summary for this stage
        total = len(stage_results)
        significant = sum(1 for r in stage_results if r['significant'])
        report_lines.append("")
        report_lines.append(f"Summary: {significant}/{total} models show Model1 significantly better ({significant/total*100:.1f}%)")
        report_lines.append("")

    # Overall summary
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 40)

    total_comparisons = 0
    significant_comparisons = 0

    for stage in ['onset', 'escalation']:
        if stage in all_delong_results:
            stage_comparisons = len(all_delong_results[stage])
            stage_significant = sum(1 for r in all_delong_results[stage].values() if r['significant'])
            total_comparisons += stage_comparisons
            significant_comparisons += stage_significant

            report_lines.append(f"{stage.capitalize()}: {stage_significant}/{stage_comparisons} ({stage_significant/stage_comparisons*100:.1f}%)")

    report_lines.append(f"Overall: {significant_comparisons}/{total_comparisons} ({significant_comparisons/total_comparisons*100:.1f}%)")
    report_lines.append("")
    report_lines.append("*** p < 0.05 (significant)")
    report_lines.append("")
    report_lines.append("=" * 80)

    report_path = os.path.join(output_dir, f"onesided_delong_test_report_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return report_path

def create_pairwise_all_models_report(pairwise_comparison_data, timestamp, output_dir='results/reports'):
    """
    Creates a text report summarizing pairwise DeLong test results for all AI models within each stage.

    Args:
        pairwise_comparison_data (list): List of dictionaries, each containing results from pairwise comparisons.
        timestamp (str): Timestamp for naming the file.
        output_dir (str): Directory to save the report.

    Returns:
        str or None: Path to the generated report, or None if no pairwise results.
    """
    if not pairwise_comparison_data:
        print("No pairwise comparisons could be performed for report.")
        return None

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("PAIRWISE DELONG TEST RESULTS - ALL AI MODELS")
    report_lines.append("Testing: Model1 vs Model2 (two-sided, paired test)")
    report_lines.append("=" * 80)
    report_lines.append("")

    os.makedirs(output_dir, exist_ok=True)

    # Group by stage
    for stage in ['onset', 'escalation']:
        stage_results = [r for r in pairwise_comparison_data if r['stage'] == stage]
        if not stage_results:
            continue

        report_lines.append(f"{stage.upper()} STAGE")
        report_lines.append("-" * 40)

        # Add table header
        report_lines.append(f"{'Model 1':<20} {'Model 2':<20} {'AUC 1':<8} {'AUC 2':<8} {'Z-stat':<8} {'p-value':<8} {'CI lower':<10} {'CI upper':<10}")
        report_lines.append("-" * 100)

        # Sort by p-value
        stage_results.sort(key=lambda x: x['p_value'] if x['p_value'] is not None else float('inf'))

        for result in stage_results:
            significance = "***" if result['significant'] else ""
            report_lines.append(f"{result['model1']:<20} {result['model2']:<20} {result['auc1']:<8.3f} {result['auc2']:<8.3f} {result['z_statistic']:<8.3f} {result['p_value']:<8.3f} {result['ci_lower']:<10.3f} {result['ci_upper']:.3f} {significance}")

        # Summary for this stage
        total = len(stage_results)
        significant = sum(1 for r in stage_results if r['significant'])
        report_lines.append("")
        report_lines.append(f"  Summary: {significant}/{total} comparisons show significant differences ({significant/total*100:.1f}%)")
        report_lines.append("")

    # Overall summary
    total = len(pairwise_comparison_data)
    significant = sum(1 for r in pairwise_comparison_data if r['significant'])
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(f"Total comparisons: {total}")
    report_lines.append(f"Significant differences: {significant}")
    report_lines.append(f"Percentage significant: {significant/total*100:.1f}%")
    report_lines.append("")
    report_lines.append("*** p < 0.05 (significant)")
    report_lines.append("")
    report_lines.append("=" * 80)

    report_path = os.path.join(output_dir, f"pairwise_all_models_report_{timestamp}.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    return report_path

def load_optimization_history(stage, theory):
    """
    Loads the optimization history for a given stage and theory from a JSON file.

    Args:
        stage (str): The prediction stage ('onset' or 'escalation').
        theory (str): The name of the theoretical model.

    Returns:
        list or None: List of optimization records, or None if file not found/error.
    """
    json_file = f"results/raw_data_for_plots/ai_optimization_history_{stage}_{theory}.json"
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading optimization history from {json_file}: {e}")
            return None
    return None

def load_results_data(file_path='results/raw_data_for_plots/overall_model_summary.csv'):
    """
    Loads the overall model summary data from a CSV file.

    Args:
        file_path (str): Path to the overall model summary CSV file.

    Returns:
        pd.DataFrame or None: DataFrame of the summary data, or None if file not found/error.
    """
    if os.path.exists(file_path):
        try:
            summary_df = pd.read_csv(file_path)
            return summary_df
        except Exception as e:
            print(f"Error loading summary data from {file_path}: {e}")
            return None
    else:
        print(f"Summary file not found: {file_path}")
        return None

def load_all_analysis_results(theories, stages, results_base_dir='results/raw_data_for_plots'):
    """
    Loads all necessary analysis results from saved files to reconstruct the
    'all_results' dictionary structure. This is useful for regenerating reports
    or plots without re-running the full model training.

    Args:
        theories (list): List of theoretical model names.
        stages (list): List of prediction stages ('onset', 'escalation').
        results_base_dir (str): Base directory where raw data for plots is saved.

    Returns:
        dict: Reconstructed 'all_results' dictionary.
    """
    all_results = {}
    base_path = Path(results_base_dir)

    for stage in stages:
        all_results[stage] = {}
        for theory in theories:
            # Initialize result placeholders
            current_theory_result = {
                'human': {'auc': 0.5, 'feature_importance': pd.DataFrame(), 'predictions': np.array([]), 'true_labels': np.array([])},
                'ai': {'auc': 0.5, 'feature_importance': pd.DataFrame(), 'predictions': np.array([]), 'true_labels': np.array([])},
                'best_ai_params': {},
                'features': [], # This might need to be loaded from config/models.json or inferred
                'optimization_history': []
            }

            # Load human predictions and feature importance
            human_pred_file = base_path / f"predictions_human_{stage}_{theory}.csv"
            human_feat_file = base_path / f"feature_importance_human_{stage}_{theory}.csv"
            if human_pred_file.exists():
                try:
                    human_preds_df = pd.read_csv(human_pred_file)
                    current_theory_result['human']['true_labels'] = human_preds_df['true_labels'].values
                    current_theory_result['human']['predictions'] = human_preds_df['predictions'].values
                    current_theory_result['human']['auc'] = roc_auc_score(human_preds_df['true_labels'], human_preds_df['predictions'])
                except Exception as e:
                    print(f"Warning: Could not load human predictions for {stage}-{theory}: {e}")
            if human_feat_file.exists():
                try:
                    current_theory_result['human']['feature_importance'] = pd.read_csv(human_feat_file)
                except Exception as e:
                    print(f"Warning: Could not load human feature importance for {stage}-{theory}: {e}")

            # Load AI predictions and feature importance
            ai_pred_file = base_path / f"predictions_ai_{stage}_{theory}.csv"
            ai_feat_file = base_path / f"feature_importance_ai_{stage}_{theory}.csv"
            if ai_pred_file.exists():
                try:
                    ai_preds_df = pd.read_csv(ai_pred_file)
                    current_theory_result['ai']['true_labels'] = ai_preds_df['true_labels'].values
                    current_theory_result['ai']['predictions'] = ai_preds_df['predictions'].values
                    current_theory_result['ai']['auc'] = roc_auc_score(ai_preds_df['true_labels'], ai_preds_df['predictions'])
                except Exception as e:
                    print(f"Warning: Could not load AI predictions for {stage}-{theory}: {e}")
            if ai_feat_file.exists():
                try:
                    current_theory_result['ai']['feature_importance'] = pd.read_csv(ai_feat_file)
                except Exception as e:
                    print(f"Warning: Could not load AI feature importance for {stage}-{theory}: {e}")

            # Load best AI parameters and optimization history
            opt_history_file = base_path / f"ai_optimization_history_{stage}_{theory}.json"
            if opt_history_file.exists():
                try:
                    history = load_optimization_history(stage, theory) # Reuse existing helper
                    if history:
                        current_theory_result['optimization_history'] = history
                        # The last record in history usually contains the applied_params for the best model
                        # However, 'best_ai_params' in the main results is the one that yielded best validation AUC
                        # So, we should try to find that specific one if needed, or rely on the last one.
                        # For simplicity, let's just use the last one for now or the one that yielded max AUC
                        best_record = max(history, key=lambda x: x.get('validation_auc', 0))
                        current_theory_result['best_ai_params'] = best_record.get('applied_params', {})
                except Exception as e:
                    print(f"Warning: Could not load optimization history for {stage}-{theory}: {e}")

            # Note: 'features' list is not saved in raw_data_for_plots, it comes from config/models.json
            # So this field will remain empty unless explicitly loaded from config.
            # For this load function, we'll leave it as an empty list or fill it from a passed config.
            # For now, it's fine as it's primarily used in the main analysis run, not when loading existing.

            all_results[stage][theory] = current_theory_result

    return all_results
