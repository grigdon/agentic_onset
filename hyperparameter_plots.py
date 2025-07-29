import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path

def load_optimization_history(stage, theory):
    """Load the optimization history for a given stage and theory."""
    json_file = f"results/raw_data_for_plots/ai_optimization_history_{stage}_{theory}.json"
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)
    return None

def create_hyperparameter_plots():
    """Create hyperparameter optimization plots for both stages."""
    
    # Define stages and theories
    stages = ['onset', 'escalation']
    theories = ['base', 'complete', 'resource_mobilization', 'political_opportunity', 
                'grievances', 'pom_gm', 'pom_rmm', 'rmm_gm']
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hyperparameter Optimization Progress: AUC Scores Across Iterations', 
                 fontsize=16, fontweight='bold')
    
    # Colors for different theories
    colors = plt.cm.Set3(np.linspace(0, 1, len(theories)))
    
    # Plot for each stage
    for stage_idx, stage in enumerate(stages):
        for theory_idx, theory in enumerate(theories):
            # Load optimization history
            history = load_optimization_history(stage, theory)
            
            if history is not None:
                # Extract iterations and AUC scores
                iterations = [record['iteration'] for record in history]
                auc_scores = [record['validation_auc'] for record in history]
                
                # Plot line for this theory - FIX: Use correct subplot indexing
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
    
    # Customize onset plot (left column)
    axes[0, 0].set_title('Onset Stage: Validation AUC Progress', fontweight='bold')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Validation AUC')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].set_ylim(0.65, 0.85)
    
    # Customize escalation plot (right column)
    axes[0, 1].set_title('Escalation Stage: Validation AUC Progress', fontweight='bold')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Validation AUC')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].set_ylim(0.65, 0.85)
    
    # Create improvement plots (bottom row)
    for stage_idx, stage in enumerate(stages):
        improvements = []
        theory_names = []
        
        for theory in theories:
            history = load_optimization_history(stage, theory)
            if history is not None:
                # Calculate improvement from first to best iteration
                first_auc = history[0]['validation_auc']
                best_auc = max([record['validation_auc'] for record in history])
                improvement = best_auc - first_auc
                
                improvements.append(improvement)
                theory_names.append(theory.replace('_', ' ').title())
        
        # Create bar plot for improvements
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
            
            # Add value labels on bars
            for bar, improvement in zip(bars, improvements):
                height = bar.get_height()
                axes[1, stage_idx].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{improvement:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/hyperparameter_optimization_plots.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Hyperparameter optimization plots saved to: results/hyperparameter_optimization_plots.png")

def create_detailed_iteration_plots():
    """Create detailed plots showing parameter evolution across iterations."""
    
    stages = ['onset', 'escalation']
    theories = ['base', 'complete']  # Focus on key theories for detailed plots
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Detailed Hyperparameter Evolution Across Iterations', 
                 fontsize=16, fontweight='bold')
    
    for stage_idx, stage in enumerate(stages):
        for theory_idx, theory in enumerate(theories):
            history = load_optimization_history(stage, theory)
            
            if history is not None:
                # Extract parameters
                iterations = [record['iteration'] for record in history]
                n_estimators = [record['params']['n_estimators'] for record in history]
                max_depth = [record['params']['max_depth'] for record in history]
                max_features = [record['params']['max_features'] for record in history]
                
                # Create subplot
                ax = axes[stage_idx, theory_idx]
                
                # Plot multiple parameters
                ax2 = ax.twinx()
                
                # Plot n_estimators and max_depth
                line1 = ax.plot(iterations, n_estimators, 'b-o', label='n_estimators', linewidth=2)
                line2 = ax.plot(iterations, max_depth, 'r-s', label='max_depth', linewidth=2)
                
                # Plot max_features (categorical, so use secondary y-axis)
                feature_map = {'sqrt': 1, 'log2': 2, 1.0: 3}
                max_features_numeric = [feature_map.get(f, 1) for f in max_features]
                line3 = ax2.plot(iterations, max_features_numeric, 'g-^', label='max_features', linewidth=2)
                
                # Customize plot
                ax.set_title(f'{stage.title()} - {theory.replace("_", " ").title()}', fontweight='bold')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Parameter Value (n_estimators, max_depth)', color='black')
                ax2.set_ylabel('max_features (1=sqrt, 2=log2, 3=1.0)', color='green')
                ax.grid(True, alpha=0.3)
                
                # Combine legends
                lines = line1 + line2 + line3
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, loc='upper right')
                
                # Set y-axis limits - FIX: Handle None values robustly
                n_estimators_clean = [v for v in n_estimators if v is not None]
                max_depth_clean = [v for v in max_depth if v is not None]
                if n_estimators_clean and max_depth_clean:
                    max_n_est = max(n_estimators_clean)
                    max_depth_val = max(max_depth_clean)
                    ax.set_ylim(0, max(max_n_est, max_depth_val) * 1.1)
                ax2.set_ylim(0.5, 3.5)
                ax2.set_yticks([1, 2, 3])
                ax2.set_yticklabels(['sqrt', 'log2', '1.0'])
    
    plt.tight_layout()
    plt.savefig('results/detailed_parameter_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Detailed parameter evolution plots saved to: results/detailed_parameter_evolution.png")

if __name__ == "__main__":
    import numpy as np
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    print("Creating hyperparameter optimization plots...")
    create_hyperparameter_plots()
    
    print("\nCreating detailed parameter evolution plots...")
    create_detailed_iteration_plots()
    
    print("\nAll plots generated successfully!") 