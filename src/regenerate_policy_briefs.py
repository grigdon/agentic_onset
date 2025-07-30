import os
import json
import pandas as pd
from openai import OpenAI

# Load the existing results
def load_existing_results():
    """Load the existing results from the saved files."""
    all_results = {}
    
    # Load the overall summary
    summary_file = "results/raw_data_for_plots/overall_model_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        # Reconstruct the results structure
        for stage in ['onset', 'escalation']:
            all_results[stage] = {}
            for theory in ['base', 'complete', 'resource_mobilization', 'political_opportunity', 
                          'grievances', 'pom_gm', 'pom_rmm', 'rmm_gm']:
                
                # Load human predictions
                human_pred_file = f"results/raw_data_for_plots/predictions_human_{stage}_{theory}.csv"
                ai_pred_file = f"results/raw_data_for_plots/predictions_ai_{stage}_{theory}.csv"
                human_feat_file = f"results/raw_data_for_plots/feature_importance_human_{stage}_{theory}.csv"
                ai_feat_file = f"results/raw_data_for_plots/feature_importance_ai_{stage}_{theory}.csv"
                
                if (os.path.exists(human_pred_file) and os.path.exists(ai_pred_file) and 
                    os.path.exists(human_feat_file) and os.path.exists(ai_feat_file)):
                    
                    # Load predictions and calculate AUC
                    human_preds = pd.read_csv(human_pred_file)
                    ai_preds = pd.read_csv(ai_pred_file)
                    human_feat = pd.read_csv(human_feat_file)
                    ai_feat = pd.read_csv(ai_feat_file)
                    
                    # Calculate AUC (simplified - you might need to recalculate this properly)
                    from sklearn.metrics import roc_auc_score
                    human_auc = roc_auc_score(human_preds['true_labels'], human_preds['predictions'])
                    ai_auc = roc_auc_score(ai_preds['true_labels'], ai_preds['predictions'])
                    
                    # Load best AI parameters
                    opt_file = f"results/raw_data_for_plots/ai_optimization_history_{stage}_{theory}.json"
                    best_ai_params = {}
                    if os.path.exists(opt_file):
                        with open(opt_file, 'r') as f:
                            opt_history = json.load(f)
                            if opt_history:
                                best_ai_params = opt_history[-1]['applied_params']
                    
                    all_results[stage][theory] = {
                        'human': {
                            'auc': human_auc,
                            'feature_importance': human_feat
                        },
                        'ai': {
                            'auc': ai_auc,
                            'feature_importance': ai_feat
                        },
                        'best_ai_params': best_ai_params
                    }
    
    return all_results

# Theory descriptions
THEORY_DESCRIPTIONS = {
    "base": "Control variables only - demographic and temporal factors",
    "complete": "All theoretical mechanisms combined",
    "resource_mobilization": "Economic resources, terrain advantages, and mobilization capacity",
    "political_opportunity": "Political system openness, regime characteristics, and institutional factors",
    "grievances": "Group status, autonomy loss, and political exclusion",
    "pom_gm": "Political opportunities interacting with grievances",
    "pom_rmm": "Political opportunities enabling resource mobilization",
    "rmm_gm": "Resource availability amplifying grievances"
}

def generate_policy_brief_prompt(stage, theory, result, theory_description):
    """Generate a prompt for the LLM to create a policy brief."""
    
    # Extract top 5 features from AI model
    top_features = result['ai']['feature_importance'].head(5)['feature'].tolist()
    
    # Get the best AUC (AI or human, whichever is better)
    best_auc = max(result['ai']['auc'], result['human']['auc'])
    
    # Stage-specific insights
    stage_insights = {
        "onset": "Early mobilization patterns show that economic and demographic factors are critical predictors. Groups with larger populations and higher GDP per capita are more likely to mobilize, while mountainous terrain provides strategic advantages.",
        "escalation": "Escalation from nonviolent to violent conflict is strongly influenced by political factors and group grievances. Status exclusion and autonomy loss are key triggers, while political system characteristics can either facilitate or prevent escalation."
    }
    
    # Analysis recommendations
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

def create_stage_theory_policy_reports(client, all_results):
    """Generate and save policy briefs for each stage and theory using the LLM."""
    output_dir = os.path.join('results', 'policy_briefs')
    os.makedirs(output_dir, exist_ok=True)
    
    for stage, results in all_results.items():
        for theory, result in results.items():
            theory_description = THEORY_DESCRIPTIONS.get(theory, "No description available.")
            prompt = generate_policy_brief_prompt(stage, theory, result, theory_description)
            try:
                response = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                content = response.choices[0].message.content
            except Exception as e:
                content = f"[Error generating policy brief: {e}]"
            filename = os.path.join(output_dir, f"policy_brief_{stage}_{theory}.txt")
            with open(filename, 'w') as f:
                f.write(content)
            print(f"Policy brief saved to {filename}")

def main():
    # Load existing results
    print("Loading existing results...")
    all_results = load_existing_results()
    
    if not all_results:
        print("No existing results found. Please run the main analysis first.")
        return
    
    # Set up OpenAI client
    api_key = input("Enter your OpenAI API key: ")
    client = OpenAI(api_key=api_key)
    
    # Generate policy briefs
    print("Generating policy briefs with updated format...")
    create_stage_theory_policy_reports(client, all_results)
    print("Policy briefs regenerated successfully!")

if __name__ == "__main__":
    main() 