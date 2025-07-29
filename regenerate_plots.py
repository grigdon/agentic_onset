import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path

# Set much larger font sizes for better readability
plt.rcParams.update({
    'font.size': 20,           # Base font size
    'axes.titlesize': 28,      # Title font size
    'axes.labelsize': 24,      # Axis label font size
    'xtick.labelsize': 20,     # X-tick label font size
    'ytick.labelsize': 20,     # Y-tick label font size
    'legend.fontsize': 18,     # Legend font size
    'figure.titlesize': 32,    # Figure title font size
    'figure.dpi': 300,         # High DPI for better quality
})

def load_results_data():
    """Load the results data from the saved files."""
    results_dir = Path('results/raw_data_for_plots')
    
    # Load overall model summary
    summary_file = results_dir / 'overall_model_summary.csv'
    if summary_file.exists():
        summary_df = pd.read_csv(summary_file)
        return summary_df
    else:
        print(f"Summary file not found: {summary_file}")
        return None

def create_onset_escalation_plots():
    """Create onset and escalation comparison plots with large fonts."""
    
    # Load the data
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Create a large figure for better visibility
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle('AI vs Human-Tuned Random Forest Performance Comparison', 
                 fontsize=36, fontweight='bold', y=0.98)
    
    # Filter data for onset and escalation stages
    onset_data = summary_df[summary_df['stage'] == 'onset']
    escalation_data = summary_df[summary_df['stage'] == 'escalation']
    
    # Colors for better visibility
    human_color = '#2E86AB'  # Blue
    ai_color = '#A23B72'     # Purple
    improvement_color = '#F18F01'  # Orange
    
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
    
    # Add value labels on bars
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
    
    # Add value labels on bars
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
    
    # Add improvement labels
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
    
    # Add improvement labels
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
    
    plt.tight_layout(pad=3.0)  # Increase padding between subplots
    
    # Save the plot with high quality
    plot_filepath = os.path.join('results', 'onset_escalation_comparison_large_fonts.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Large font onset and escalation plots saved to: {plot_filepath}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nONSET STAGE:")
    print(f"  Total models: {len(onset_data)}")
    print(f"  AI models that beat human: {sum(1 for imp in improvements_onset if imp > 0)}/{len(improvements_onset)}")
    print(f"  Average improvement: {np.mean(improvements_onset):.4f}")
    print(f"  Best human AUC: {max(human_aucs_onset):.4f}")
    print(f"  Best AI AUC: {max(ai_aucs_onset):.4f}")
    print(f"  Max improvement: {max(improvements_onset):.4f}")
    
    print(f"\nESCALATION STAGE:")
    print(f"  Total models: {len(escalation_data)}")
    print(f"  AI models that beat human: {sum(1 for imp in improvements_escalation if imp > 0)}/{len(improvements_escalation)}")
    print(f"  Average improvement: {np.mean(improvements_escalation):.4f}")
    print(f"  Best human AUC: {max(human_aucs_escalation):.4f}")
    print(f"  Best AI AUC: {max(ai_aucs_escalation):.4f}")
    print(f"  Max improvement: {max(improvements_escalation):.4f}")

def create_focused_onset_escalation_plots():
    """Create focused onset and escalation comparison plots with maximum font sizes."""
    
    # Load the data
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Set even larger font sizes for maximum readability
    plt.rcParams.update({
        'font.size': 28,           # Base font size
        'axes.titlesize': 36,      # Title font size
        'axes.labelsize': 32,      # Axis label font size
        'xtick.labelsize': 24,     # X-tick label font size
        'ytick.labelsize': 24,     # Y-tick label font size
        'legend.fontsize': 26,     # Legend font size
        'figure.titlesize': 40,    # Figure title font size
        'figure.dpi': 300,         # High DPI for better quality
    })
    
    # Create a very large figure for maximum visibility
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(32, 16))
    fig.suptitle('AI vs Human-Tuned Random Forest Performance: Onset vs Escalation Stages', 
                 fontsize=44, fontweight='bold', y=0.98)
    
    # Filter data for onset and escalation stages
    onset_data = summary_df[summary_df['stage'] == 'onset']
    escalation_data = summary_df[summary_df['stage'] == 'escalation']
    
    # Colors for better visibility
    human_color = '#2E86AB'  # Blue
    ai_color = '#A23B72'     # Purple
    
    # Plot 1: Onset Stage - AUC Comparison
    theories_onset = [t.replace('_', ' ').title() for t in onset_data['theory']]
    human_aucs_onset = onset_data['human_auc'].values
    ai_aucs_onset = onset_data['ai_auc'].values
    
    x = np.arange(len(theories_onset))
    width = 0.4  # Slightly wider bars
    
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
    
    # Add value labels on bars with larger fonts
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
    
    # Add value labels on bars with larger fonts
    for bar in bars3:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    for bar in bars4:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=22, fontweight='bold')
    
    plt.tight_layout(pad=2.0)  # Increase padding between subplots
    
    # Save the plot with high quality
    plot_filepath = os.path.join('results', 'onset_escalation_focused_large_fonts.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Focused large font onset and escalation plots saved to: {plot_filepath}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("FOCUSED PLOT SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nONSET STAGE:")
    print(f"  Total models: {len(onset_data)}")
    print(f"  AI models that beat human: {sum(1 for imp in (ai_aucs_onset - human_aucs_onset) if imp > 0)}/{len(onset_data)}")
    print(f"  Average improvement: {np.mean(ai_aucs_onset - human_aucs_onset):.4f}")
    print(f"  Best human AUC: {max(human_aucs_onset):.4f}")
    print(f"  Best AI AUC: {max(ai_aucs_onset):.4f}")
    print(f"  Max improvement: {max(ai_aucs_onset - human_aucs_onset):.4f}")
    
    print(f"\nESCALATION STAGE:")
    print(f"  Total models: {len(escalation_data)}")
    print(f"  AI models that beat human: {sum(1 for imp in (ai_aucs_escalation - human_aucs_escalation) if imp > 0)}/{len(escalation_data)}")
    print(f"  Average improvement: {np.mean(ai_aucs_escalation - human_aucs_escalation):.4f}")
    print(f"  Best human AUC: {max(human_aucs_escalation):.4f}")
    print(f"  Best AI AUC: {max(ai_aucs_escalation):.4f}")
    print(f"  Max improvement: {max(ai_aucs_escalation - human_aucs_escalation):.4f}")

def create_variable_importance_plots():
    """Create variable importance plots for AI-tuned models in both stages."""
    
    # Load the data
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Set large font sizes for better readability
    plt.rcParams.update({
        'font.size': 24,           # Base font size
        'axes.titlesize': 32,      # Title font size
        'axes.labelsize': 28,      # Axis label font size
        'xtick.labelsize': 20,     # X-tick label font size
        'ytick.labelsize': 20,     # Y-tick label font size
        'legend.fontsize': 22,     # Legend font size
        'figure.titlesize': 36,    # Figure title font size
        'figure.dpi': 300,         # High DPI for better quality
    })
    
    # Load feature importance data
    results_dir = Path('results/raw_data_for_plots')
    
    # Create a large figure for better visibility - only 2 plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Variable Importance: Best AI-Tuned Random Forest Models', 
                 fontsize=40, fontweight='bold', y=0.98)
    
    # Colors for different theories
    colors = plt.cm.Set3(np.linspace(0, 1, 8))
    
    # Plot for each stage
    for stage_idx, stage in enumerate(['onset', 'escalation']):
        stage_data = summary_df[summary_df['stage'] == stage]
        
        # Get the best AI model for this stage
        best_ai_idx = stage_data['ai_auc'].idxmax()
        best_theory = stage_data.loc[best_ai_idx, 'theory']
        
        # Load feature importance for the best AI model
        feature_importance_file = results_dir / f'feature_importance_ai_{stage}_{best_theory}.csv'
        
        if feature_importance_file.exists():
            feature_importance_df = pd.read_csv(feature_importance_file)
            
            # Get top 10 features
            top_features = feature_importance_df.head(10)
            
            # Create horizontal bar plot
            ax = ax1 if stage == 'onset' else ax2
            
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
            
            # Add value labels on bars
            for i, (bar, importance) in enumerate(zip(bars, top_features['importance'])):
                ax.text(importance + 0.003, i, f'{importance:.3f}', 
                       ha='left', va='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout(pad=3.0)  # Increase padding between subplots
    
    # Save the plot with high quality
    plot_filepath = os.path.join('results', 'variable_importance_ai_models.png')
    plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Variable importance plots saved to: {plot_filepath}")

def create_separate_auc_plots():
    """Create separate AUC comparison plots with large fonts for LaTeX."""
    
    # Load the data
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Set very large font sizes for LaTeX compatibility
    plt.rcParams.update({
        'font.size': 28,           # Base font size
        'axes.titlesize': 36,      # Title font size
        'axes.labelsize': 32,      # Axis label font size
        'xtick.labelsize': 24,     # X-tick label font size
        'ytick.labelsize': 24,     # Y-tick label font size
        'legend.fontsize': 26,     # Legend font size
        'figure.titlesize': 40,    # Figure title font size
        'figure.dpi': 300,         # High DPI for better quality
    })
    
    # Filter data for onset and escalation stages
    onset_data = summary_df[summary_df['stage'] == 'onset']
    escalation_data = summary_df[summary_df['stage'] == 'escalation']
    
    # Colors for better visibility
    human_color = '#2E86AB'  # Blue
    ai_color = '#A23B72'     # Purple
    
    # Create separate plots for each stage
    for stage, stage_data in [('onset', onset_data), ('escalation', escalation_data)]:
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        theories = [t.replace('_', ' ').title() for t in stage_data['theory']]
        human_aucs = stage_data['human_auc'].values
        ai_aucs = stage_data['ai_auc'].values
        
        x = np.arange(len(theories))
        width = 0.4  # Slightly wider bars
        
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
        
        # Add value labels on bars with larger fonts
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=20, fontweight='bold')
        
        plt.tight_layout(pad=2.0)
        
        # Save the plot with high quality
        plot_filepath = os.path.join('results', f'{stage}_auc_comparison_latex.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"{stage.title()} AUC comparison plot saved to: {plot_filepath}")

def create_separate_improvement_plots():
    """Create separate improvement plots with large fonts for LaTeX."""
    
    # Load the data
    summary_df = load_results_data()
    if summary_df is None:
        print("Could not load summary data. Please run main.py first to generate the data.")
        return
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Set very large font sizes for LaTeX compatibility
    plt.rcParams.update({
        'font.size': 28,           # Base font size
        'axes.titlesize': 36,      # Title font size
        'axes.labelsize': 32,      # Axis label font size
        'xtick.labelsize': 24,     # X-tick label font size
        'ytick.labelsize': 24,     # Y-tick label font size
        'legend.fontsize': 26,     # Legend font size
        'figure.titlesize': 40,    # Figure title font size
        'figure.dpi': 300,         # High DPI for better quality
    })
    
    # Filter data for onset and escalation stages
    onset_data = summary_df[summary_df['stage'] == 'onset']
    escalation_data = summary_df[summary_df['stage'] == 'escalation']
    
    # Create separate plots for each stage
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
        
        # Add improvement labels
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
        
        # Save the plot with high quality
        plot_filepath = os.path.join('results', f'{stage}_improvement_latex.png')
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"{stage.title()} improvement plot saved to: {plot_filepath}")

def run_improvement_plots_only():
    """Run only the improvement plots generation."""
    print("Generating separate improvement plots for LaTeX...")
    create_separate_improvement_plots()
    print("Done!")

if __name__ == "__main__":
    print("Generating onset and escalation plots with large fonts...")
    create_onset_escalation_plots()
    
    print("\nGenerating focused onset and escalation plots with maximum font sizes...")
    create_focused_onset_escalation_plots()
    
    print("\nGenerating variable importance plots for AI-tuned models...")
    create_variable_importance_plots()
    
    print("\nGenerating separate AUC comparison plots for LaTeX...")
    create_separate_auc_plots()
    
    print("\nGenerating separate improvement plots for LaTeX...")
    create_separate_improvement_plots()
    
    print("Done!") 