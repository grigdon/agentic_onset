import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from pathlib import Path

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
        ax.set_ylabel('AUC Improvement (AI - Human)', fontsize=24, fontweight='bold')  # Smaller y-axis title
        # Removed title for cleaner look
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

if __name__ == "__main__":
    print("Generating separate improvement plots for LaTeX...")
    create_separate_improvement_plots()
    print("Done!") 