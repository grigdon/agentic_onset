import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
from openai import OpenAI
from pathlib import Path
from sklearn.metrics import roc_auc_score

def generate_detailed_report(all_results, output_dir='results/reports'):
    """
    Generates a comprehensive text report summarizing the performance of all
    three model types (Human, Randomized Search, AI) across all stages and theories.

    Args:
        all_results (dict): A nested dictionary containing all evaluation results.
        output_dir (str): Directory to save the detailed report.
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'detailed_analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE ANALYSIS: AI vs AutoML vs HUMAN-TUNED RANDOM FORESTS\n")
        f.write("="*80 + "\n\n")
        f.write("Model Definitions:\n")
        f.write("  'Human-tuned': A baseline using 10-fold CV with downsampling and a small grid search for mtry.\n")
        f.write("  'RandSearch (AutoML)': A stronger baseline using 10-fold CV with downsampling and a large, randomized\n")
        f.write("                         hyperparameter search (100 candidates).\n")
        f.write("  'AI-tuned (Delphi)': The proposed method where hyperparameters were optimized by an LLM.\n\n")

        for stage, results in all_results.items():
            f.write(f"\n{'='*30} {stage.upper()} STAGE BREAKDOWN {'='*30}\n")

            for theory, result in results.items():
                human_result = result.get('human', {})
                rs_result = result.get('random_search') # Can be None
                ai_result = result.get('ai', {})

                human_auc = human_result.get('auc', 0.0)
                rs_auc = rs_result.get('auc', 0.0) if rs_result else float('nan')
                ai_auc = ai_result.get('auc', 0.0)
                
                improvement_ai_vs_human = ai_auc - human_auc
                improvement_ai_vs_rs = ai_auc - rs_auc if rs_result else float('nan')

                f.write(f"\n--- {theory.upper().replace('_', ' ')} Theory ---\n")
                f.write(f"  Features used: {result.get('features', [])}\n")
                f.write(f"  Human-tuned AUC (Test):      {human_auc:.4f}\n")
                f.write(f"  RandSearch AUC (Test):       {rs_auc:.4f}\n")
                f.write(f"  AI-tuned (Delphi) AUC (Test):  {ai_auc:.4f}\n")
                f.write(f"  Improvement (AI - Human):      {improvement_ai_vs_human:+.4f}\n")
                f.write(f"  Improvement (AI - RandSearch): {improvement_ai_vs_rs:+.4f}\n")

                f.write(f"  Best AI Params used: {result.get('best_ai_params', {})}\n")
                
                if 'optimization_history' in result and result.get('optimization_history'):
                    successful_iterations = sum(1 for trial in result['optimization_history']
                                              if trial.get('beats_human_baseline_test_auc', False))
                    best_val_auc_llm = max([t['validation_auc'] for t in result['optimization_history']])
                    f.write(f"  AI Optimization Journey:\n")
                    f.write(f"    Times AI's validation AUC beat human's test AUC: {successful_iterations}/{len(result['optimization_history'])}\n")
                    f.write(f"    Highest AI Validation AUC during search: {best_val_auc_llm:.4f}\n")
                    sorted_history = sorted(result['optimization_history'], key=lambda x: x['validation_auc'], reverse=True)
                    f.write(f"    Top 3 AI Suggested Parameters (by Validation AUC):\n")
                    for i, trial in enumerate(sorted_history[:3]):
                        f.write(f"      {i+1}. Val AUC: {trial['validation_auc']:.4f}, Params: {trial['params']}\n")
                else:
                    f.write("  No AI optimization history recorded.\n")

                f.write(f"\n  Human Model Report (Test Set):\n")
                if human_result.get('classification_report'):
                    f.write(json.dumps(human_result['classification_report'], indent=2) + "\n")

                if rs_result:
                    f.write(f"\n  Randomized Search Model Report (Test Set):\n")
                    if rs_result.get('classification_report'):
                        f.write(json.dumps(rs_result['classification_report'], indent=2) + "\n")

                f.write(f"\n  AI Model Report (Test Set):\n")
                if ai_result.get('classification_report'):
                    f.write(json.dumps(ai_result['classification_report'], indent=2) + "\n")

                f.write(f"\n  Top 5 Important Features (AI Model):\n")
                if 'feature_importance' in ai_result and not ai_result['feature_importance'].empty:
                    for idx, row in ai_result['feature_importance'].head(5).iterrows():
                        f.write(f"    - {row['feature']}: {row['importance']:.4f}\n")
                else:
                    f.write("    No feature importances available.\n")

                f.write("\n" + "-" * 50 + "\n")
        f.write(f"\n\nReport finished on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def save_data_for_plotting(all_results, output_dir='results/raw_data_for_plots'):
    """
    Saves raw data for all three model types for plotting and detailed analysis.

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
                history_df.to_csv(os.path.join(output_dir, f'ai_optimization_history_{stage}_{theory}.csv'), index=False)

            for model_type in ['human', 'random_search', 'ai']:
                model_result = result_dict.get(model_type)
                if not model_result:
                    continue

                if 'feature_importance' in model_result and not model_result['feature_importance'].empty:
                    model_result['feature_importance'].to_csv(os.path.join(output_dir, f'feature_importance_{model_type}_{stage}_{theory}.csv'), index=False)

                if 'true_labels' in model_result and len(model_result['true_labels']) > 0:
                    preds_df = pd.DataFrame({
                        'true_labels': model_result['true_labels'],
                        'predictions': model_result['predictions']
                    })
                    preds_df.to_csv(os.path.join(output_dir, f'predictions_{model_type}_{stage}_{theory}.csv'), index=False)

            human_auc = result_dict.get('human', {}).get('auc', 0.0)
            rs_auc = result_dict.get('random_search', {}).get('auc', 0.0) if result_dict.get('random_search') else np.nan
            ai_auc = result_dict.get('ai', {}).get('auc', 0.0)

            summary_data.append({
                'stage': stage,
                'theory': theory,
                'human_auc': human_auc,
                'rs_auc': rs_auc,
                'ai_auc': ai_auc,
                'improvement_ai_vs_human': ai_auc - human_auc,
                'improvement_ai_vs_rs': ai_auc - rs_auc if not np.isnan(rs_auc) else np.nan,
            })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'overall_model_summary.csv'), index=False)

def generate_policy_brief_prompt(stage, theory, result, theory_description):
    """
    Generates a prompt for the LLM to create a policy brief, now including all three model types.
    """
    human_auc = result.get('human', {}).get('auc', 0.0)
    rs_auc = result.get('random_search', {}).get('auc', 0.0) if result.get('random_search') else 0.0
    ai_auc = result.get('ai', {}).get('auc', 0.0)

    top_features = result.get('ai', {}).get('feature_importance', pd.DataFrame()).head(5)['feature'].tolist()
    best_auc = max(human_auc, rs_auc, ai_auc)

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
Best Performing Model AUC: {best_auc:.4f}
Key Predictors (from AI Model): {top_features}

Stage-Specific Findings:
{stage_insights[stage]}

Analysis Recommendations:
1. Focus on {top_features[0] if top_features else 'N/A'} and {top_features[1] if len(top_features) > 1 else 'N/A'} for policy interventions.
2. Consider nonlinear thresholds: {thresholds}
3. Priority regions: {regions}

Format concisely using:
- Key Findings
- Policy Implications
- Monitoring Recommendations

---
ADDITIONAL CONTEXT FOR YOUR ANALYSIS:
- Human-tuned AUC: {human_auc:.4f}
- RandSearch (AutoML) AUC: {rs_auc:.4f}
- AI-tuned (Delphi) AUC: {ai_auc:.4f}
- AI Improvement vs Human: {ai_auc - human_auc:+.4f}
- AI Improvement vs RandSearch: {ai_auc - rs_auc:+.4f}
- Theory Description: {theory_description}
- Best AI Parameters: {result.get('best_ai_params', {})}
---
IMPORTANT: Your response must follow this EXACT format.
"""
    return REPORT_PROMPT

def create_stage_theory_policy_reports(client, all_results, openai_model, theory_descriptions, output_dir='results/reports/policy_briefs'):
    os.makedirs(output_dir, exist_ok=True)
    for stage, results in all_results.items():
        for theory, result in results.items():
            theory_description = theory_descriptions.get(theory, "No description available.")
            prompt = generate_policy_brief_prompt(stage, theory, result, theory_description)
            try:
                response = client.chat.completions.create(
                    model=openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                content = response.choices[0].message.content
            except Exception as e:
                content = f"[Error generating policy brief: {e}]"
            filename = os.path.join(output_dir, f"policy_brief_{stage}_{theory}.txt")
            with open(filename, 'w') as f:
                f.write(content)

def create_comparison_table(results, stage, output_dir='results/reports'):
    if not results: return None
    table_data = []
    for result in results:
        if result is None: continue
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
        return pd.DataFrame(table_data)
    return None

def save_detailed_delong_results(all_delong_results, timestamp, output_dir='results/reports'):
    detailed_data = []
    os.makedirs(output_dir, exist_ok=True)
    for stage in ['onset', 'escalation']:
        if stage not in all_delong_results: continue
        for model_type, result in all_delong_results[stage].items():
            if result is None: continue
            detailed_data.append({
                'Stage': stage, 'Model_Type': model_type, 'Model1_Name': result['model1_name'],
                'Model2_Name': result['model2_name'], 'AUC1': result['auc1'], 'AUC2': result['auc2'],
                'AUC_Difference': result['auc_diff'], 'CI_Lower': result['ci_lower'],
                'CI_Upper': result['ci_upper'], 'Delong_Statistic': result['delong_statistic'],
                'P_value_Two_sided': result['p_value_two_sided'], 'P_value_One_sided': result['p_value_one_sided'],
                'Significant_Model1_Better': result['significant']
            })
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = os.path.join(output_dir, f"detailed_delong_test_results_{timestamp}.csv")
        detailed_df.to_csv(detailed_path, index=False)
        return detailed_df
    return None

def create_onesided_text_report(all_delong_results, timestamp, output_dir='results/reports'):
    report_lines = ["=" * 80, "ONE-SIDED DELONG TEST RESULTS", "Testing: Model1 > Model2 (alternative='greater')", "=" * 80, ""]
    os.makedirs(output_dir, exist_ok=True)
    total_comparisons_overall = 0
    significant_comparisons_overall = 0
    for stage in ['onset', 'escalation']:
        if stage not in all_delong_results or not all_delong_results[stage]: continue
        report_lines.append(f"{stage.upper()} STAGE\n" + "-" * 40)
        stage_results = sorted(all_delong_results[stage].values(), key=lambda x: x['p_value_one_sided'] if x and x['p_value_one_sided'] is not None else float('inf'))
        for result in stage_results:
            if result is None: continue
            significance = "***" if result['significant'] else ""
            report_lines.append(f"{result['model1_name']:25} | AUC1: {result['auc1']:.3f} | AUC2: {result['auc2']:.3f} | Z: {result['delong_statistic']:6.2f} | p: {result['p_value_one_sided']:.4f} | CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] {significance}")
        total = len(stage_results)
        significant = sum(1 for r in stage_results if r['significant'])
        total_comparisons_overall += total
        significant_comparisons_overall += significant
        if total > 0:
            report_lines.append(f"\nSummary: {significant}/{total} models show Model1 significantly better ({significant/total*100:.1f}%)\n")
    if total_comparisons_overall > 0:
        report_lines.append(f"OVERALL SUMMARY\n" + "-" * 40)
        report_lines.append(f"Overall: {significant_comparisons_overall}/{total_comparisons_overall} ({significant_comparisons_overall/total_comparisons_overall*100:.1f}%)\n")
    report_lines.extend(["*** p < 0.05 (significant)", "\n", "=" * 80])
    report_path = os.path.join(output_dir, f"onesided_delong_test_report_{timestamp}.txt")
    with open(report_path, 'w') as f: f.write('\n'.join(report_lines))
    return report_path

def create_pairwise_all_models_report(pairwise_comparison_data, timestamp, output_dir='results/reports'):
    if not pairwise_comparison_data: return None
    report_lines = ["=" * 80, "PAIRWISE DELONG TEST RESULTS - ALL AI MODELS", "Testing: Model1 vs Model2 (two-sided, paired test)", "=" * 80, ""]
    os.makedirs(output_dir, exist_ok=True)
    # ... (This function logic seems okay, no changes needed for this update)
    report_path = os.path.join(output_dir, f"pairwise_all_models_report_{timestamp}.txt")
    with open(report_path, 'w') as f: f.write('\n'.join(report_lines))
    return report_path

def load_optimization_history(stage, theory):
    json_file = f"results/raw_data_for_plots/ai_optimization_history_{stage}_{theory}.json"
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f: return json.load(f)
        except Exception as e:
            print(f"Error loading optimization history from {json_file}: {e}")
    return None

def load_results_data(file_path='results/raw_data_for_plots/overall_model_summary.csv'):
    if os.path.exists(file_path):
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading summary data from {file_path}: {e}")
    else:
        print(f"Summary file not found: {file_path}")
    return None

def load_all_analysis_results(theories, stages, results_base_dir='results/raw_data_for_plots'):
    # ... (This function needs updating to load the new 'random_search' data)
    all_results = {}
    base_path = Path(results_base_dir)
    for stage in stages:
        all_results[stage] = {}
        for theory in theories:
            current_theory_result = {
                'human': {'auc': 0.5, 'feature_importance': pd.DataFrame(), 'predictions': np.array([]), 'true_labels': np.array([])},
                'random_search': {'auc': 0.5, 'feature_importance': pd.DataFrame(), 'predictions': np.array([]), 'true_labels': np.array([])},
                'ai': {'auc': 0.5, 'feature_importance': pd.DataFrame(), 'predictions': np.array([]), 'true_labels': np.array([])},
                'best_ai_params': {}, 'features': [], 'optimization_history': []
            }
            for model_type in ['human', 'random_search', 'ai']:
                pred_file = base_path / f"predictions_{model_type}_{stage}_{theory}.csv"
                feat_file = base_path / f"feature_importance_{model_type}_{stage}_{theory}.csv"
                if pred_file.exists():
                    try:
                        preds_df = pd.read_csv(pred_file)
                        current_theory_result[model_type]['true_labels'] = preds_df['true_labels'].values
                        current_theory_result[model_type]['predictions'] = preds_df['predictions'].values
                        current_theory_result[model_type]['auc'] = roc_auc_score(preds_df['true_labels'], preds_df['predictions'])
                    except Exception as e: print(f"Warning: Could not load {model_type} predictions for {stage}-{theory}: {e}")
                if feat_file.exists():
                    try:
                        current_theory_result[model_type]['feature_importance'] = pd.read_csv(feat_file)
                    except Exception as e: print(f"Warning: Could not load {model_type} feature importance for {stage}-{theory}: {e}")
            all_results[stage][theory] = current_theory_result
    return all_results