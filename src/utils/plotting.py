import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve
from pathlib import Path

# Import load_results_data from reporting module
from src.utils.reporting import load_results_data

def plot_roc_curve(y_true, y_preds_human, y_preds_ai, theory, stage, human_auc, ai_auc, output_dir='results/plots'):
    """
    Generates and saves an ROC curve plot comparing human-tuned and AI-tuned models.

    Args:
        y_true (np.array): True binary labels.
        y_preds_human (np.array): Predicted probabilities from the human-tuned model.
        y_preds_ai (np.array): Predicted probabilities from the AI-tuned model.
        theory (str): Name of the theoretical model.
        stage (str): Prediction stage ('onset' or 'escalation').
        human_auc (float): AUC score for the human-tuned model.
        ai_auc (float): AUC score for the AI-tuned model.
        output_dir (str): Directory to save the plot.
    """
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

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'roc_curve_{stage}_{theory}.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_plots(results_dict, stage, output_dir='results/plots'):
    """
    Generates and saves summary comparison plots (AUC, improvement, distribution, top features).

    Args:
        results_dict (dict): Dictionary containing all model evaluation results for a stage.
        stage (str): Prediction stage ('onset' or 'escalation').
        output_dir (str): Directory to save the plots.
    """
    os.makedirs(output_dir, exist_ok=True)

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
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Adjusted legend position
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
        if best_theory_idx < len(theories):
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
                ax4.set_title('No feature importance to plot for best AI model.')
                ax4.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        else:
            ax4.set_title('No AI models to plot feature importance (index issue).')
            ax4.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
    else:
        ax4.set_title('No AI models to plot feature importance.')
        ax4.text(0.5, 0.5, 'No data', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

    plt.tight_layout()
    plot_filepath = os.path.join(output_dir, f'{stage}_comparison_plots.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
    plt.close()

# --- Hyperparameter Optimization Plots (Moved from original hyperparameter_plots.py) ---

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

def create_onset_escalation_plots(output_dir='results/plots'):
    """
    Create onset and escalation comparison plots with large fonts for general viewing.

    Args:
        output_dir (str): Directory to save the plots.
    """
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Set much larger font sizes for better readability
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 28,
        'axes.labelsize': 24,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 18,
        'figure.titlesize': 32,
        'figure.dpi': 300,
    })

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('AI vs Human-Tuned Random Forest Performance Comparison',
                 fontsize=36, fontweight='bold', y=0.98)

    onset_data = summary_df[summary_df['stage'] == 'onset']
    escalation_data = summary_df[summary_df['stage'] == 'escalation']

    human_color = '#2E86AB'
    ai_color = '#A23B72'
    improvement_color = '#F18F01'

    # Plot 1: Onset Stage - AUC Comparison
    theories_onset = [t.replace('_', ' ').title() for t in onset_data['theory']]
    human_aucs_onset = onset_data['human_auc'].values
    ai_aucs_onset = onset_data['ai_auc'].values

    x = np.arange(len(theories_onset))
    width = 0.35

    bars1 = ax1.bar(x - width/2, human_aucs_onset, width, label='Human-Tuned',
                    color=human_color, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, ai_aucs_onset, width, label='AI-Tuned',
                    color=ai_color, alpha=0.8, edgecolor='black', linewidth=1)

    ax1.set_xlabel('Theoretical Models', fontsize=28, fontweight='bold')
    ax1.set_ylabel('AUC Score (Test Set)', fontsize=28, fontweight='bold')
    ax1.set_title('Onset Stage: AUC Comparison', fontsize=32, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(theories_onset, rotation=45, ha='right', fontsize=18)
    ax1.legend(fontsize=22, loc='upper right')
    ax1.grid(True, alpha=0.3, linewidth=1.5)
    ax1.set_ylim(0.5, 1.0)
    ax1.tick_params(axis='both', which='major', labelsize=20)

    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Plot 2: Escalation Stage - AUC Comparison
    theories_escalation = [t.replace('_', ' ').title() for t in escalation_data['theory']]
    human_aucs_escalation = escalation_data['human_auc'].values
    ai_aucs_escalation = escalation_data['ai_auc'].values

    x = np.arange(len(theories_escalation))

    bars3 = ax2.bar(x - width/2, human_aucs_escalation, width, label='Human-Tuned',
                    color=human_color, alpha=0.8, edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + width/2, ai_aucs_escalation, width, label='AI-Tuned',
                    color=ai_color, alpha=0.8, edgecolor='black', linewidth=1)

    ax2.set_xlabel('Theoretical Models', fontsize=28, fontweight='bold')
    ax2.set_ylabel('AUC Score (Test Set)', fontsize=28, fontweight='bold')
    ax2.set_title('Escalation Stage: AUC Comparison', fontsize=32, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(theories_escalation, rotation=45, ha='right', fontsize=18)
    ax2.legend(fontsize=22, loc='upper right')
    ax2.grid(True, alpha=0.3, linewidth=1.5)
    ax2.set_ylim(0.5, 1.0)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Plot 3: Onset Stage - Improvements
    improvements_onset = ai_aucs_onset - human_aucs_onset
    colors_onset = ['green' if imp > 0 else 'red' for imp in improvements_onset]

    bars5 = ax3.bar(theories_onset, improvements_onset, color=colors_onset, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Theoretical Models', fontsize=28, fontweight='bold')
    ax3.set_ylabel('AUC Improvement (AI - Human)', fontsize=24, fontweight='bold')
    ax3.set_title('Onset: AI vs Human Performance', fontsize=32, fontweight='bold')
    ax3.set_xticklabels(theories_onset, rotation=45, ha='right', fontsize=18)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
    ax3.grid(True, alpha=0.3, linewidth=1.5)
    ax3.tick_params(axis='both', which='major', labelsize=20)

    for bar, imp in zip(bars5, improvements_onset):
        height = bar.get_height()
        if height >= 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                    f'{imp:+.3f}', ha='center', va='bottom',
                    fontsize=18, fontweight='bold')
        else:
            ax3.text(bar.get_x() + bar.get_width()/2., height - 0.015,
                    f'{imp:+.3f}', ha='center', va='top',
                    fontsize=18, fontweight='bold')

    # Plot 4: Escalation Stage - Improvements
    improvements_escalation = ai_aucs_escalation - human_aucs_escalation
    colors_escalation = ['green' if imp > 0 else 'red' for imp in improvements_escalation]

    bars6 = ax4.bar(theories_escalation, improvements_escalation, color=colors_escalation, alpha=0.8,
                    edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Theoretical Models', fontsize=28, fontweight='bold')
    ax4.set_ylabel('AUC Improvement (AI - Human)', fontsize=24, fontweight='bold')
    ax4.set_title('Escalation: AI vs Human Performance', fontsize=32, fontweight='bold')
    ax4.set_xticklabels(theories_escalation, rotation=45, ha='right', fontsize=18)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=2)
    ax4.grid(True, alpha=0.3, linewidth=1.5)
    ax4.tick_params(axis='both', which='major', labelsize=20)

    for bar, imp in zip(bars6, improvements_escalation):
        height = bar.get_height()
        if height >= 0:
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                    f'{imp:+.3f}', ha='center', va='bottom',
                    fontsize=18, fontweight='bold')
        else:
            ax4.text(bar.get_x() + bar.get_width()/2., height - 0.015,
                    f'{imp:+.3f}', ha='center', va='top',
                    fontsize=18, fontweight='bold')

    plt.tight_layout(pad=3.0)
    plot_filepath = os.path.join(output_dir, 'onset_escalation_comparison_large_fonts.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() # Use plt.close()

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

def create_separate_auc_plots(output_dir='results/plots'):
    """
    Create separate AUC comparison plots with large fonts for LaTeX.

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

    human_color = '#2E86AB'
    ai_color = '#A23B72'

    for stage, stage_data in [('onset', onset_data), ('escalation', escalation_data)]:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))

        theories = [t.replace('_', ' ').title() for t in stage_data['theory']]
        human_aucs = stage_data['human_auc'].values
        ai_aucs = stage_data['ai_auc'].values

        x = np.arange(len(theories))
        width = 0.4

        bars1 = ax.bar(x - width/2, human_aucs, width, label='Human-Tuned',
                      color=human_color, alpha=0.9, edgecolor='black', linewidth=2)
        bars2 = ax.bar(x + width/2, ai_aucs, width, label='AI-Tuned',
                      color=ai_color, alpha=0.9, edgecolor='black', linewidth=2)

        ax.set_xlabel('Theoretical Models', fontsize=32, fontweight='bold')
        ax.set_ylabel('AUC Score (Test Set)', fontsize=32, fontweight='bold')
        ax.set_title(f'{stage.title()} Stage: AUC Comparison', fontsize=36, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(theories, rotation=45, ha='right', fontsize=24)
        ax.legend(fontsize=26, loc='upper right')
        ax.grid(True, alpha=0.4, linewidth=2)
        ax.set_ylim(0.5, 1.0)
        ax.tick_params(axis='both', which='major', labelsize=24)

        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')

        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')

        plt.tight_layout(pad=2.0)

        plot_filepath = os.path.join(output_dir, f'{stage}_auc_comparison_latex.png')
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
