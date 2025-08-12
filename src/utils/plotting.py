import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from pathlib import Path

from src.utils.reporting import load_results_data

def plot_roc_curve(y_true, predictions, labels, aucs, theory, stage, output_dir='results/plots'):
    """
    Generates and saves an ROC curve plot comparing multiple models.

    Args:
        y_true (np.array): True binary labels.
        predictions (list of np.array): List of predicted probabilities for each model.
        labels (list of str): List of labels for each model.
        aucs (list of float): List of AUC scores for each model.
        theory (str): Name of the theoretical model.
        stage (str): Prediction stage ('onset' or 'escalation').
        output_dir (str): Directory to save the plot.
    """
    plt.figure(figsize=(8, 7))
    colors = ['#2E86AB', '#F18F01', '#A23B72']  # Human, RandSearch, AI

    for i, (preds, label, auc) in enumerate(zip(predictions, labels, aucs)):
        if preds is not None and len(preds) > 0:
            fpr, tpr, _ = roc_curve(y_true, preds)
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, label=f'{label} (AUC = {auc:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {theory.replace("_", " ").title()} - {stage.capitalize()} Stage')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'roc_curve_{stage}_{theory}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def create_focused_onset_escalation_plots(output_dir='results/plots'):
    """
    Creates focused comparison plots for Onset and Escalation stages,
    now showing three models: Human-Tuned, RandSearch, and AI-Tuned.

    Args:
        output_dir (str): Directory to save the plots.
    """
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 28, 'axes.titlesize': 36, 'axes.labelsize': 32,
        'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 22,
        'figure.titlesize': 40, 'figure.dpi': 300,
    })

    human_color = '#2E86AB'
    rs_color = '#F18F01'
    ai_color = '#A23B72'

    for stage in ['onset', 'escalation']:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        stage_data = summary_df[summary_df['stage'] == stage].copy()
        stage_data.fillna(0, inplace=True)

        theories = [t.replace('_', ' ').title() for t in stage_data['theory']]
        x = np.arange(len(theories))
        width = 0.25

        ax.bar(x - width, stage_data['human_auc'], width, label='Human-Tuned', color=human_color, alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.bar(x, stage_data['rs_auc'], width, label='RandSearch (AutoML)', color=rs_color, alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.bar(x + width, stage_data['ai_auc'], width, label='AI-Tuned (Delphi)', color=ai_color, alpha=0.9, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Theoretical Models', fontsize=32, fontweight='bold')
        ax.set_ylabel('AUC Score (Test Set)', fontsize=32, fontweight='bold')
        ax.set_title(f'{stage.title()} Stage: Model Performance Comparison', fontsize=36, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(theories, rotation=45, ha='right', fontsize=24)
        ax.legend(fontsize=22)
        ax.grid(True, which='major', axis='y', alpha=0.4, linewidth=2)
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis='both', which='major', labelsize=24)
        
        plt.tight_layout(pad=2.0)
        plot_filepath = os.path.join(output_dir, f'auc_comparison_{stage}_3way.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def create_separate_improvement_plots(output_dir='results/plots'):
    """
    Creates separate improvement plots, showing AI performance against both baselines.

    Args:
        output_dir (str): Directory to save the plots.
    """
    summary_df = load_results_data()
    if summary_df is None: return

    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 28, 'axes.titlesize': 36, 'axes.labelsize': 32,
        'xtick.labelsize': 24, 'ytick.labelsize': 24, 'legend.fontsize': 22,
    })

    for stage in ['onset', 'escalation']:
        fig, ax = plt.subplots(1, 1, figsize=(20, 12))
        stage_data = summary_df[summary_df['stage'] == stage].copy()
        stage_data.fillna(0, inplace=True)
        
        theories = [t.replace('_', ' ').title() for t in stage_data['theory']]
        x = np.arange(len(theories))
        width = 0.35

        imp_vs_human = stage_data['improvement_ai_vs_human']
        imp_vs_rs = stage_data['improvement_ai_vs_rs']
        
        ax.bar(x - width/2, imp_vs_human, width, label='AI vs. Human-Tuned', color='#A23B72', alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.bar(x + width/2, imp_vs_rs, width, label='AI vs. RandSearch', color='#F18F01', alpha=0.9, edgecolor='black', linewidth=1.5)

        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=3)
        ax.set_ylabel('AI (Delphi) AUC Improvement', fontsize=32, fontweight='bold')
        ax.set_title(f'{stage.title()} Stage: AI (Delphi) Improvement Over Baselines', fontsize=36, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(theories, rotation=45, ha='right', fontsize=24)
        ax.legend(fontsize=22)
        ax.grid(True, which='major', axis='y', alpha=0.4, linewidth=2)
        
        plt.tight_layout(pad=2.0)
        plot_filepath = os.path.join(output_dir, f'improvement_comparison_{stage}_3way.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

def create_hyperparameter_optimization_plots(theories, stages, output_dir='results/plots', load_history_func=None):
    """
    Creates hyperparameter optimization plots showing AUC scores across iterations for each theory and stage.

    Args:
        theories (list): List of theoretical model names.
        stages (list): List of prediction stages ('onset', 'escalation').
        output_dir (str): Directory to save the plots.
        load_history_func (function): Function to load optimization history (e.g., from src.utils.reporting).
    """
    if load_history_func is None:
        raise ValueError("load_history_func must be provided to create_hyperparameter_optimization_plots.")

    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Optimization Progress: AUC Scores Across Iterations',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.Set3(np.linspace(0, 1, len(theories)))

    for stage_idx, stage in enumerate(stages):
        for theory_idx, theory in enumerate(theories):
            history = load_history_func(stage, theory)

            if history is not None:
                iterations = [record['iteration'] for record in history]
                auc_scores = [record['validation_auc'] for record in history]

                if stage == 'onset':
                    axes[0, 0].plot(iterations, auc_scores,
                                   marker='o', linewidth=2, markersize=6,
                                   label=theory.replace('_', ' ').title(),
                                   color=colors[theory_idx])
                else:  # escalation
                    axes[0, 1].plot(iterations, auc_scores,
                                   marker='o', linewidth=2, markersize=6,
                                   label=theory.replace('_', ' ').title(),
                                   color=colors[theory_idx])

        # Customize plot for current stage
        if stage == 'onset':
            ax = axes[0, 0]
            ax.set_title('Onset Stage: Validation AUC Progress', fontweight='bold')
        else:
            ax = axes[0, 1]
            ax.set_title('Escalation Stage: Validation AUC Progress', fontweight='bold')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Validation AUC')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0.65, 0.85) # Fixed y-axis limits for consistency

    # Create improvement plots (bottom row)
    for stage_idx, stage in enumerate(stages):
        improvements = []
        theory_names = []

        for theory in theories:
            history = load_history_func(stage, theory)
            if history is not None:
                first_auc = history[0]['validation_auc']
                best_auc = max([record['validation_auc'] for record in history])
                improvement = best_auc - first_auc

                improvements.append(improvement)
                theory_names.append(theory.replace('_', ' ').title())

        if improvements:
            bars = axes[1, stage_idx].bar(range(len(improvements)), improvements,
                                         color=colors[:len(improvements)], alpha=0.7)
            axes[1, stage_idx].set_title(f'{stage.title()} Stage: AUC Improvements',
                                        fontweight='bold')
            axes[1, stage_idx].set_xlabel('Theory')
            axes[1, stage_idx].set_ylabel('AUC Improvement (Best - First)')
            axes[1, stage_idx].set_xticks(range(len(theory_names)))
            axes[1, stage_idx].set_xticklabels(theory_names, rotation=45, ha='right')
            axes[1, stage_idx].grid(True, alpha=0.3, axis='y')

            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                axes[1, stage_idx].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{improvement:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plot_filepath = os.path.join(output_dir, 'hyperparameter_optimization_plots.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close() # Use plt.close() instead of plt.show() for non-interactive environments

def create_detailed_parameter_evolution_plots(theories, stages, output_dir='results/plots', load_history_func=None):
    """
    Creates detailed plots showing parameter evolution across iterations for specific theories.

    Args:
        theories (list): List of theoretical model names to plot in detail.
        stages (list): List of prediction stages ('onset', 'escalation').
        output_dir (str): Directory to save the plots.
        load_history_func (function): Function to load optimization history (e.g., from src.utils.reporting).
    """
    if load_history_func is None:
        raise ValueError("load_history_func must be provided to create_detailed_parameter_evolution_plots.")

    os.makedirs(output_dir, exist_ok=True)

    num_theories_to_plot = len(theories)
    num_stages_to_plot = len(stages)

    if num_theories_to_plot * num_stages_to_plot > 4:
        print("Warning: Too many detailed plots requested for a 2x2 grid. Adjusting to plot only first 2 theories.")
        theories = theories[:2]

    fig, axes = plt.subplots(num_stages_to_plot, num_theories_to_plot, figsize=(8 * num_theories_to_plot, 6 * num_stages_to_plot))
    if num_stages_to_plot == 1 and num_theories_to_plot == 1:
        axes = np.array([[axes]])
    elif num_stages_to_plot == 1 or num_theories_to_plot == 1:
        axes = axes.reshape(num_stages_to_plot, num_theories_to_plot)


    fig.suptitle('Detailed Hyperparameter Evolution Across Iterations',
                 fontsize=16, fontweight='bold')

    for stage_idx, stage in enumerate(stages):
        for theory_idx, theory in enumerate(theories):
            if theory_idx >= len(theories):
                continue

            history = load_history_func(stage, theory)

            ax = axes[stage_idx, theory_idx]

            if history is not None:
                iterations = [record['iteration'] for record in history]
                n_estimators = [record['params'].get('n_estimators') for record in history]
                max_depth = [record['params'].get('max_depth') for record in history]
                max_features = [record['params'].get('max_features') for record in history]
                class_weight = [record['params'].get('class_weight') for record in history]


                ax2 = ax.twinx()

                line1 = ax.plot(iterations, n_estimators, 'b-o', label='n_estimators', linewidth=2)
                line2 = ax.plot(iterations, max_depth, 'r-s', label='max_depth', linewidth=2)

                feature_map = {'sqrt': 1, 'log2': 2, 1.0: 3, None: 0}
                max_features_numeric = [feature_map.get(f, 0) for f in max_features]
                line3 = ax2.plot(iterations, max_features_numeric, 'g-^', label='max_features', linewidth=2)

                class_weight_labels = [cw if cw is not None else 'None' for cw in class_weight]
                if len(set(class_weight_labels)) > 1:
                    ax.text(0.02, 0.98, f"Class Weight: {', '.join(list(set(class_weight_labels)))}",
                            transform=ax.transAxes, fontsize=8, verticalalignment='top')

                ax.set_title(f'{stage.title()} - {theory.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Parameter Value (n_estimators, max_depth)', color='black')
                ax2.set_ylabel('max_features (0=None, 1=sqrt, 2=log2, 3=1.0)', color='green')
                ax.grid(True, alpha=0.3)

                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper right')

                n_estimators_clean = [v for v in n_estimators if v is not None]
                max_depth_clean = [v for v in max_depth if v is not None]

                min_y1 = 0
                max_y1 = 100
                if n_estimators_clean:
                    max_y1 = max(max_y1, max(n_estimators_clean) * 1.1)
                if max_depth_clean:
                    max_y1 = max(max_y1, max(max_depth_clean) * 1.1)
                ax.set_ylim(min_y1, max_y1)


                ax2.set_ylim(-0.5, 3.5)
                ax2.set_yticks([0, 1, 2, 3])
                ax2.set_yticklabels(['None', 'sqrt', 'log2', '1.0'])
            else:
                ax.set_title(f'{stage.title()} - {theory.replace("_", " ").title()} (No History)', fontweight='bold')
                ax.text(0.5, 0.5, 'No optimization history found.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


    plt.tight_layout()
    plot_filepath = os.path.join(output_dir, 'detailed_parameter_evolution.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

def create_focused_onset_escalation_plots(output_dir='results/plots'):
    """
    Create focused onset and escalation comparison plots with maximum font sizes.

    Args:
        output_dir (str): Directory to save the plots.
    """
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 36,
        'axes.labelsize': 32,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 26,
        'figure.titlesize': 40,
        'figure.dpi': 300,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
    fig.suptitle('AI vs Human-Tuned Random Forest Performance: Onset vs Escalation Stages',
                 fontsize=44, fontweight='bold', y=0.98)

    onset_data = summary_df[summary_df['stage'] == 'onset']
    escalation_data = summary_df[summary_df['stage'] == 'escalation']

    human_color = '#2E86AB'
    ai_color = '#A23B72'

    # Plot 1: Onset Stage - AUC Comparison
    theories_onset = [t.replace('_', ' ').title() for t in onset_data['theory']]
    human_aucs_onset = onset_data['human_auc'].values
    ai_aucs_onset = onset_data['ai_auc'].values

    x = np.arange(len(theories_onset))
    width = 0.4

    bars1 = ax1.bar(x - width/2, human_aucs_onset, width, label='Human-Tuned',
                    color=human_color, alpha=0.9, edgecolor='black', linewidth=2)
    bars2 = ax1.bar(x + width/2, ai_aucs_onset, width, label='AI-Tuned',
                    color=ai_color, alpha=0.9, edgecolor='black', linewidth=2)

    ax1.set_xlabel('Theoretical Models', fontsize=36, fontweight='bold')
    ax1.set_ylabel('AUC Score (Test Set)', fontsize=36, fontweight='bold')
    ax1.set_title('Onset Stage: AUC Comparison', fontsize=40, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(theories_onset, rotation=45, ha='right', fontsize=24)
    ax1.legend(fontsize=28, loc='upper right')
    ax1.grid(True, alpha=0.4, linewidth=2)
    ax1.set_ylim(0.5, 1.0)
    ax1.tick_params(axis='both', which='major', labelsize=24)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

    # Plot 2: Escalation Stage - AUC Comparison
    theories_escalation = [t.replace('_', ' ').title() for t in escalation_data['theory']]
    human_aucs_escalation = escalation_data['human_auc'].values
    ai_aucs_escalation = escalation_data['ai_auc'].values

    x = np.arange(len(theories_escalation))

    bars3 = ax2.bar(x - width/2, human_aucs_escalation, width, label='Human-Tuned',
                    color=human_color, alpha=0.9, edgecolor='black', linewidth=2)
    bars4 = ax2.bar(x + width/2, ai_aucs_escalation, width, label='AI-Tuned',
                    color=ai_color, alpha=0.9, edgecolor='black', linewidth=2)

    ax2.set_xlabel('Theoretical Models', fontsize=36, fontweight='bold')
    ax2.set_ylabel('AUC Score (Test Set)', fontsize=36, fontweight='bold')
    ax2.set_title('Escalation Stage: AUC Comparison', fontsize=40, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(theories_escalation, rotation=45, ha='right', fontsize=24)
    ax2.legend(fontsize=28, loc='upper right')
    ax2.grid(True, alpha=0.4, linewidth=2)
    ax2.set_ylim(0.5, 1.0)
    ax2.tick_params(axis='both', which='major', labelsize=24)

    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')

    plt.tight_layout(pad=2.0)
    plot_filepath = os.path.join(output_dir, 'onset_escalation_focused_large_fonts.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() # Use plt.close()

def create_variable_importance_plots(output_dir='results/plots'):
    """
    Create variable importance plots for AI-tuned models in both stages.

    Args:
        output_dir (str): Directory to save the plots.
    """
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 24,
        'axes.titlesize': 32,
        'axes.labelsize': 28,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 22,
        'figure.titlesize': 36,
        'figure.dpi': 300,
    })

    results_dir = Path('results/raw_data_for_plots')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Variable Importance: Best AI-Tuned Random Forest Models',
                 fontsize=40, fontweight='bold', y=0.98)

    colors = plt.cm.Set3(np.linspace(0, 1, 8))

    for stage_idx, stage in enumerate(['onset', 'escalation']):
        stage_data = summary_df[summary_df['stage'] == stage]

        if not stage_data.empty: # Ensure there's data for the stage
            best_ai_idx = stage_data['ai_auc'].idxmax()
            best_theory = stage_data.loc[best_ai_idx, 'theory']

            feature_importance_file = results_dir / f'feature_importance_ai_{stage}_{best_theory}.csv'

            ax = ax1 if stage == 'onset' else ax2

            if feature_importance_file.exists():
                feature_importance_df = pd.read_csv(feature_importance_file)

                top_features = feature_importance_df.head(10)

                if not top_features.empty:
                    bars = ax.barh(range(len(top_features)), top_features['importance'],
                                  color=colors[stage_idx], alpha=0.8, edgecolor='black', linewidth=2)

                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features['feature'], fontsize=18)
                    ax.set_xlabel('Feature Importance', fontsize=28, fontweight='bold')
                    ax.set_title(f'{stage.title()} Stage: Top Features\n({best_theory.replace("_", " ").title()})',
                                fontsize=32, fontweight='bold')
                    ax.invert_yaxis()
                    ax.grid(True, alpha=0.3, linewidth=2)
                    ax.tick_params(axis='both', which='major', labelsize=20)

                    for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                        ax.text(importance + 0.003, i, f'{importance:.3f}',
                               ha='left', va='center', fontsize=16, fontweight='bold')
                else:
                    ax.set_title(f'{stage.title()} Stage: No Top Features\n({best_theory.replace("_", " ").title()})')
                    ax.text(0.5, 0.5, 'No feature importance data.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            else:
                ax.set_title(f'{stage.title()} Stage: No Feature Importance File\n({best_theory.replace("_", " ").title()})')
                ax.text(0.5, 0.5, 'File not found.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            ax = ax1 if stage == 'onset' else ax2
            ax.set_title(f'{stage.title()} Stage: No Data Available')
            ax.text(0.5, 0.5, 'No summary data for this stage.', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


    plt.tight_layout(pad=3.0)
    plot_filepath = os.path.join(output_dir, 'variable_importance_ai_models.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() # Use plt.close()


def create_separate_improvement_plots(output_dir='results/plots'):
    """
    Create separate improvement plots with large fonts for LaTeX.

    Args:
        output_dir (str): Directory to save the plots.
    """
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return

    os.makedirs(output_dir, exist_ok=True)

    plt.rcParams.update({
        'font.size': 28,
        'axes.titlesize': 36,
        'axes.labelsize': 32,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 26,
        'figure.titlesize': 40,
        'figure.dpi': 300,
    })

    onset_data = summary_df[summary_df['stage'] == 'onset']
    escalation_data = summary_df[summary_df['stage'] == 'escalation']

    for stage, stage_data in [('onset', onset_data), ('escalation', escalation_data)]:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        theories = [t.replace('_', ' ').title() for t in stage_data['theory']]
        human_aucs = stage_data['human_auc'].values
        ai_aucs = stage_data['ai_auc'].values
        improvements = ai_aucs - human_aucs
        colors = ['green' if imp > 0 else 'red' for imp in improvements]

        bars = ax.bar(theories, improvements, color=colors, alpha=0.8,
                     edgecolor='black', linewidth=2)

        ax.set_xlabel('Theoretical Models', fontsize=32, fontweight='bold')
        ax.set_ylabel('AUC Improvement (AI - Human)', fontsize=24, fontweight='bold')
        ax.set_xticklabels(theories, rotation=45, ha='right', fontsize=24)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=3)
        ax.grid(True, alpha=0.4, linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=24)

        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            if height >= 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{imp:+.3f}', ha='center', va='bottom',
                       fontsize=20, fontweight='bold')
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height - 0.02,
                       f'{imp:+.3f}', ha='center', va='top',
                       fontsize=20, fontweight='bold')

        plt.tight_layout(pad=2.0)

        plot_filepath = os.path.join(output_dir, f'{stage}_improvement_latex.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close() # Use plt.close()
