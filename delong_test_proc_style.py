import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def load_prediction_data(file_path):
    """Load prediction data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df['true_labels'].values, df['predictions'].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def delong_test_proc_style(y_true, y_pred1, y_pred2, debug_label=None):
    """
    Delong test implementation following pROC R package logic
    This is a more direct implementation of the original Delong paper
    """
    try:
        # Calculate ROC curves and AUCs
        fpr1, tpr1, _ = roc_curve(y_true, y_pred1)
        fpr2, tpr2, _ = roc_curve(y_true, y_pred2)
        auc1 = auc(fpr1, tpr1)
        auc2 = auc(fpr2, tpr2)
        
        # Calculate the difference in AUC
        auc_diff = auc1 - auc2
        
        # Get unique prediction values for both models
        unique_pred1 = np.unique(y_pred1)
        unique_pred2 = np.unique(y_pred2)
        
        # Calculate the empirical influence functions
        n = len(y_true)
        n_pos = np.sum(y_true == 1)
        n_neg = n - n_pos
        
        # Initialize influence functions
        v10 = np.zeros(n)
        v01 = np.zeros(n)
        
        # Calculate influence functions for each observation
        for i in range(n):
            if y_true[i] == 1:  # Positive case
                v10[i] = np.sum(y_pred1[y_true == 0] < y_pred1[i]) / n_neg
                v01[i] = np.sum(y_pred2[y_true == 0] < y_pred2[i]) / n_neg
            else:  # Negative case
                v10[i] = np.sum(y_pred1[y_true == 1] > y_pred1[i]) / n_pos
                v01[i] = np.sum(y_pred2[y_true == 1] > y_pred2[i]) / n_pos
        
        # Calculate the variance-covariance matrix
        v_diff = v10 - v01
        var_diff = np.var(v_diff) / n
        
        # Debug output for base model
        if debug_label == 'base':
            print(f"[DEBUG - base] auc_diff: {auc_diff}")
            print(f"[DEBUG - base] var_diff: {var_diff}")
            print(f"[DEBUG - base] v10[:5]: {v10[:5]}")
            print(f"[DEBUG - base] v01[:5]: {v01[:5]}")
            print(f"[DEBUG - base] v_diff[:5]: {v_diff[:5]}")
            print(f"[DEBUG - base] std(v_diff): {np.std(v_diff)}")
            print(f"[DEBUG - base] n: {n}")
        
        # Check for numerical issues
        if var_diff <= 0:
            print(f"Warning: Non-positive variance detected: {var_diff}")
            return None, None, None, None, None
        
        if np.isnan(var_diff) or np.isinf(var_diff):
            print(f"Warning: Invalid variance detected: {var_diff}")
            return None, None, None, None, None
        
        # Calculate the z-statistic
        z_stat = auc_diff / np.sqrt(var_diff)
        
        # Check for extreme z-statistics
        if abs(z_stat) > 100:
            print(f"Warning: Extreme z-statistic detected: {z_stat}")
            print(f"  auc_diff: {auc_diff}")
            print(f"  var_diff: {var_diff}")
            print(f"  sqrt(var_diff): {np.sqrt(var_diff)}")
        
        # Calculate p-values
        p_value_two_sided = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        p_value_one_sided = 1 - stats.norm.cdf(z_stat)
        
        # Check for numerical precision issues
        if p_value_two_sided < 1e-16:
            p_value_two_sided = 0.0
        if p_value_one_sided < 1e-16:
            p_value_one_sided = 0.0
        if p_value_one_sided > 1 - 1e-16:
            p_value_one_sided = 1.0
        
        return z_stat, p_value_two_sided, p_value_one_sided, auc1, auc2
        
    except Exception as e:
        print(f"Error in pROC-style Delong test: {e}")
        return None, None, None, None, None

def perform_proc_style_delong_test(y_true, y_pred_ai, y_pred_human, ai_model_name, human_model_name):
    """
    Perform pROC-style Delong test: AI > Human
    """
    try:
        # Calculate ROC curves
        fpr_ai, tpr_ai, _ = roc_curve(y_true, y_pred_ai)
        fpr_human, tpr_human, _ = roc_curve(y_true, y_pred_human)
        
        # Calculate AUCs
        auc_ai = auc(fpr_ai, tpr_ai)
        auc_human = auc(fpr_human, tpr_human)
        
        # Always test AI > Human (following pROC logic)
        debug_label = None
        if 'base' in ai_model_name.lower():
            debug_label = 'base'
        z_score, p_value_two_sided, p_value_one_sided, _, _ = delong_test_proc_style(y_true, y_pred_ai, y_pred_human, debug_label=debug_label)
        
        # Calculate AUC difference (AI - Human)
        auc_diff = auc_ai - auc_human
        
        # Determine significance: AI is significantly better if p < 0.05
        significant = p_value_one_sided < 0.05
        
        # Calculate confidence interval for AUC difference (approximate)
        if z_score is not None and z_score != 0:
            se_diff = abs(auc_diff / z_score)
            ci_lower = auc_diff - 1.96 * se_diff
            ci_upper = auc_diff + 1.96 * se_diff
        else:
            ci_lower = ci_upper = auc_diff
        
        return {
            'ai_model': ai_model_name,
            'human_model': human_model_name,
            'auc_ai': auc_ai,
            'auc_human': auc_human,
            'auc_diff': auc_diff,  # AI - Human (positive when AI is better)
            'delong_statistic': z_score,
            'p_value_two_sided': p_value_two_sided,
            'p_value_one_sided': p_value_one_sided,
            'significant': significant,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'fpr_ai': fpr_ai,
            'tpr_ai': tpr_ai,
            'fpr_human': fpr_human,
            'tpr_human': tpr_human
        }
    except Exception as e:
        print(f"Error in pROC-style Delong test for {ai_model_name} vs {human_model_name}: {e}")
        return None

def create_comparison_table(results, stage):
    """Create a comparison table of results."""
    if not results:
        return None
    
    table_data = []
    for result in results:
        if result is None:
            continue
        
        table_data.append({
            'AI Model': result['ai_model'],
            'Human Model': result['human_model'],
            'AI AUC': f"{result['auc_ai']:.3f}",
            'Human AUC': f"{result['auc_human']:.3f}",
            'AUC Difference (AI-Human)': f"{result['auc_diff']:+.3f}",
            '95% CI Lower': f"{result['ci_lower']:.3f}",
            '95% CI Upper': f"{result['ci_upper']:.3f}",
            'Delong Statistic': f"{result['delong_statistic']:.3f}" if result['delong_statistic'] is not None else 'N/A',
            'P-value (One-sided)': f"{result['p_value_one_sided']:.4f}" if result['p_value_one_sided'] is not None else 'N/A'
        })
    
    if table_data:
        df = pd.DataFrame(table_data)
        print(f"\n{'='*120}")
        print(f"PROC-STYLE DELONG TEST RESULTS: {stage.upper()}")
        print(f"Testing: AI > Human (alternative='greater')")
        print(f"{'='*120}")
        print(df.to_string(index=False))
        return df
    return None

def save_detailed_results(all_results, timestamp):
    """Save detailed results to a comprehensive CSV file."""
    detailed_data = []
    
    for stage in ['onset', 'escalation']:
        if stage not in all_results:
            continue
            
        for model_type, result in all_results[stage].items():
            if result is None:
                continue
                
            detailed_data.append({
                'Stage': stage,
                'Model_Type': model_type,
                'AI_Model': result['ai_model'],
                'Human_Model': result['human_model'],
                'AI_AUC': result['auc_ai'],
                'Human_AUC': result['auc_human'],
                'AUC_Difference': result['auc_diff'],
                'CI_Lower': result['ci_lower'],
                'CI_Upper': result['ci_upper'],
                'Delong_Statistic': result['delong_statistic'],
                'P_value_Two_sided': result['p_value_two_sided'],
                'P_value_One_sided': result['p_value_one_sided'],
                'Significant_AI_Better': result['significant']
            })
    
    if detailed_data:
        detailed_df = pd.DataFrame(detailed_data)
        detailed_path = f"results/detailed_delong_test_results_{timestamp}.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"\nDetailed results saved to: {detailed_path}")
        return detailed_df
    return None

def create_onesided_text_report(all_results, timestamp):
    """Create a text report with only one-sided Delong test results."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ONE-SIDED DELONG TEST RESULTS")
    report_lines.append("Testing: AI > Human (alternative='greater')")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    for stage in ['onset', 'escalation']:
        if stage not in all_results:
            continue
            
        report_lines.append(f"{stage.upper()} STAGE")
        report_lines.append("-" * 40)
        
        stage_results = []
        for model_type, result in all_results[stage].items():
            if result is None:
                continue
                
            stage_results.append({
                'model_type': model_type,
                'ai_auc': result['auc_ai'],
                'human_auc': result['auc_human'],
                'auc_diff': result['auc_diff'],
                'delong_stat': result['delong_statistic'],
                'p_value': result['p_value_one_sided'],
                'ci_lower': result['ci_lower'],
                'ci_upper': result['ci_upper'],
                'significant': result['significant']
            })
        
        # Sort by p-value (most significant first)
        stage_results.sort(key=lambda x: x['p_value'])
        
        for result in stage_results:
            significance = "***" if result['significant'] else ""
            report_lines.append(f"{result['model_type']:25} | AI: {result['ai_auc']:.3f} | Human: {result['human_auc']:.3f} | Z: {result['delong_stat']:6.2f} | p: {result['p_value']:.4f} | CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] {significance}")
        
        # Summary for this stage
        total = len(stage_results)
        significant = sum(1 for r in stage_results if r['significant'])
        report_lines.append("")
        report_lines.append(f"Summary: {significant}/{total} models show AI significantly better ({significant/total*100:.1f}%)")
        report_lines.append("")
    
    # Overall summary
    report_lines.append("OVERALL SUMMARY")
    report_lines.append("-" * 40)
    
    total_comparisons = 0
    significant_comparisons = 0
    
    for stage in ['onset', 'escalation']:
        if stage in all_results:
            stage_comparisons = len(all_results[stage])
            stage_significant = sum(1 for r in all_results[stage].values() if r['significant'])
            total_comparisons += stage_comparisons
            significant_comparisons += stage_significant
            
            report_lines.append(f"{stage.capitalize()}: {stage_significant}/{stage_comparisons} ({stage_significant/stage_comparisons*100:.1f}%)")
    
    report_lines.append(f"Overall: {significant_comparisons}/{total_comparisons} ({significant_comparisons/total_comparisons*100:.1f}%)")
    report_lines.append("")
    report_lines.append("*** p < 0.05 (significant)")
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save to file
    report_path = f"results/onesided_delong_test_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"One-sided results report saved to: {report_path}")
    
    # Also print to console
    print('\n'.join(report_lines))
    
    return report_path

def create_pairwise_best_models_test(all_results, timestamp):
    """Create pairwise Delong test for all AI models within each stage."""
    
    pairwise_results = []
    
    for stage in ['onset', 'escalation']:
        if stage not in all_results:
            continue
            
        # Get all AI models for this stage and sort by AUC
        stage_models = []
        for model_type, result in all_results[stage].items():
            if result is None:
                continue
            stage_models.append({
                'model_type': model_type,
                'auc': result['auc_ai'],
                'result': result
            })
        
        # Sort by AUC (highest first)
        stage_models.sort(key=lambda x: x['auc'], reverse=True)
        
        if len(stage_models) < 2:
            print(f"Need at least 2 models for {stage} stage pairwise testing")
            continue
        
        print(f"\n{stage.upper()} STAGE - All Pairwise Comparisons:")
        for i, model in enumerate(stage_models):
            print(f"{i+1}. {model['model_type']} (AUC: {model['auc']:.3f})")
        
        # Load the actual prediction data for all models
        results_dir = Path("results/raw_data_for_plots")
        model_predictions = {}
        
        # Load predictions for all models
        for model_info in stage_models:
            model_type = model_info['model_type']
            ai_file = results_dir / f"predictions_ai_{stage}_{model_type}.csv"
            
            if not ai_file.exists():
                print(f"File not found: {ai_file}")
                continue
                
            y_true, y_pred = load_prediction_data(ai_file)
            if y_true is not None and y_pred is not None:
                model_predictions[model_type] = {
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'auc': model_info['auc']
                }
        
        if len(model_predictions) < 2:
            print(f"Not enough valid models for {stage} stage pairwise testing")
            continue
        
        # Perform all pairwise comparisons
        model_types = list(model_predictions.keys())
        
        for i in range(len(model_types)):
            for j in range(i + 1, len(model_types)):
                model1_type = model_types[i]
                model2_type = model_types[j]
                
                # Get predictions
                pred1 = model_predictions[model1_type]
                pred2 = model_predictions[model2_type]
                
                # Verify same number of samples
                if len(pred1['y_pred']) != len(pred2['y_pred']):
                    print(f"Sample size mismatch: {model1_type} ({len(pred1['y_pred'])}) vs {model2_type} ({len(pred2['y_pred'])})")
                    continue
                
                # Perform pairwise Delong test: Model1 vs Model2
                debug_pairwise = (stage == 'onset' and model1_type == 'grievances' and model2_type == 'complete')
                z_score, p_value_two_sided, p_value_one_sided, auc1, auc2 = delong_test_proc_style(
                    pred1['y_true'], pred1['y_pred'], pred2['y_pred'], 
                    debug_label='pairwise' if debug_pairwise else None
                )
                
                if z_score is not None:
                    # Calculate confidence interval
                    auc_diff = auc1 - auc2
                    if z_score != 0:
                        se_diff = abs(auc_diff / z_score)
                        ci_lower = auc_diff - 1.96 * se_diff
                        ci_upper = auc_diff + 1.96 * se_diff
                    else:
                        ci_lower = ci_upper = auc_diff
                    
                    pairwise_results.append({
                        'stage': stage,
                        'model1': model1_type,
                        'model2': model2_type,
                        'auc1': auc1,
                        'auc2': auc2,
                        'auc_diff': auc_diff,
                        'z_statistic': z_score,
                        'p_value': p_value_two_sided,  # Use two-sided p-value like R
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'significant': p_value_two_sided < 0.05
                    })
    
    # Create pairwise report
    if pairwise_results:
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PAIRWISE DELONG TEST RESULTS - ALL AI MODELS")
        report_lines.append("Testing: Model1 vs Model2 (two-sided, paired test)")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Group by stage
        for stage in ['onset', 'escalation']:
            stage_results = [r for r in pairwise_results if r['stage'] == stage]
            if not stage_results:
                continue
                
            report_lines.append(f"{stage.upper()} STAGE")
            report_lines.append("-" * 40)
            
            # Add table header
            report_lines.append(f"{'Model 1':<20} {'Model 2':<20} {'AUC 1':<8} {'AUC 2':<8} {'Z-stat':<8} {'p-value':<8} {'CI lower':<10} {'CI upper':<10}")
            report_lines.append("-" * 100)
            
            # Sort by p-value
            stage_results.sort(key=lambda x: x['p_value'])
            
            for result in stage_results:
                significance = "***" if result['significant'] else ""
                report_lines.append(f"{result['model1']:<20} {result['model2']:<20} {result['auc1']:<8.3f} {result['auc2']:<8.3f} {result['z_statistic']:<8.3f} {result['p_value']:<8.3f} {result['ci_lower']:<10.3f} {result['ci_upper']:<10.3f} {significance}")
            
            # Summary for this stage
            total = len(stage_results)
            significant = sum(1 for r in stage_results if r['significant'])
            report_lines.append("")
            report_lines.append(f"  Summary: {significant}/{total} comparisons show significant differences ({significant/total*100:.1f}%)")
            report_lines.append("")
        
        # Overall summary
        total = len(pairwise_results)
        significant = sum(1 for r in pairwise_results if r['significant'])
        report_lines.append("OVERALL SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total comparisons: {total}")
        report_lines.append(f"Significant differences: {significant}")
        report_lines.append(f"Percentage significant: {significant/total*100:.1f}%")
        report_lines.append("")
        report_lines.append("*** p < 0.05 (significant)")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save to file
        report_path = f"results/pairwise_all_models_report_{timestamp}.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Pairwise all models report saved to: {report_path}")
        print('\n'.join(report_lines))
        
        return report_path
    else:
        print("No pairwise comparisons could be performed")
        return None

def main():
    """Main function to perform pROC-style Delong tests."""
    # Define paths
    results_dir = Path("results/raw_data_for_plots")
    
    # Define stages and model types
    stages = ['onset', 'escalation']
    model_types = ['base', 'complete', 'grievances', 'political_opportunity', 
                   'resource_mobilization', 'pom_gm', 'pom_rmm', 'rmm_gm']
    
    all_results = {}
    
    for stage in stages:
        all_results[stage] = {}
        
        for model_type in model_types:
            print(f"\nProcessing {stage} - {model_type}...")
            
            # File paths
            ai_file = results_dir / f"predictions_ai_{stage}_{model_type}.csv"
            human_file = results_dir / f"predictions_human_{stage}_{model_type}.csv"
            
            # Check if files exist
            if not ai_file.exists() or not human_file.exists():
                print(f"Files not found for {stage} - {model_type}")
                continue
            
            # Load data
            y_true_ai, y_pred_ai = load_prediction_data(ai_file)
            y_true_human, y_pred_human = load_prediction_data(human_file)
            
            if y_true_ai is None or y_true_human is None:
                continue
            
            # Verify same true labels
            if not np.array_equal(y_true_ai, y_true_human):
                print(f"Warning: True labels don't match for {stage} - {model_type}")
                continue
            
            # Perform pROC-style Delong test
            result = perform_proc_style_delong_test(
                y_true_ai, y_pred_ai, y_pred_human, 
                f"AI_{model_type}", f"Human_{model_type}"
            )
            
            if result:
                all_results[stage][model_type] = result
                print(f"✓ Completed pROC-style Delong test for {stage} - {model_type}")
                print(f"  AI AUC: {result['auc_ai']:.3f}, Human AUC: {result['auc_human']:.3f}")
                print(f"  AUC Difference (AI-Human): {result['auc_diff']:+.3f}")
                print(f"  P-value (one-sided): {result['p_value_one_sided']:.4f}")
                print(f"  AI significantly better: {result['significant']}")
    
    # Generate results
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    
    for stage in stages:
        if not all_results[stage]:
            continue
            
        # Create comparison table
        results_list = list(all_results[stage].values())
        table_df = create_comparison_table(results_list, stage)
        
        # Save table
        if table_df is not None:
            table_path = f"results/delong_test_proc_style_{stage}_{timestamp}.csv"
            table_df.to_csv(table_path, index=False)
            print(f"Table saved to: {table_path}")
    
    # Create overall summary
    print(f"\n{'='*120}")
    print("OVERALL SUMMARY - PROC-STYLE DELONG TEST")
    print("Testing: AI > Human (alternative='greater')")
    print(f"{'='*120}")
    
    total_comparisons = 0
    significant_comparisons = 0
    
    for stage in stages:
        stage_comparisons = len(all_results[stage])
        stage_significant = sum(1 for r in all_results[stage].values() if r['significant'])
        
        print(f"\n{stage.upper()}:")
        print(f"  Total comparisons: {stage_comparisons}")
        print(f"  AI significantly better: {stage_significant}")
        print(f"  Percentage AI better: {stage_significant/stage_comparisons*100:.1f}%" if stage_comparisons > 0 else "  No comparisons")
        
        total_comparisons += stage_comparisons
        significant_comparisons += stage_significant
    
    print(f"\nOVERALL:")
    print(f"  Total comparisons: {total_comparisons}")
    print(f"  AI significantly better: {significant_comparisons}")
    print(f"  Percentage AI better: {significant_comparisons/total_comparisons*100:.1f}%" if total_comparisons > 0 else "  No comparisons")

    # Save detailed results
    save_detailed_results(all_results, timestamp)

    # Create one-sided text report
    create_onesided_text_report(all_results, timestamp)

    # Create pairwise best models test
    create_pairwise_best_models_test(all_results, timestamp)

if __name__ == "__main__":
    main() 