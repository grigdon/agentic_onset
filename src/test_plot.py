import json
import os

def test_load_data():
    """Test loading escalation data to see what's happening."""
    
    # Test escalation base
    json_file = "results/raw_data_for_plots/ai_optimization_history_escalation_base.json"
    print(f"Checking file: {json_file}")
    print(f"File exists: {os.path.exists(json_file)}")
    
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data)}")
        
        if len(data) > 0:
            print(f"First record keys: {list(data[0].keys())}")
            print(f"First record validation_auc: {data[0]['validation_auc']}")
            
            # Extract all iterations and AUC scores
            iterations = [record['iteration'] for record in data]
            auc_scores = [record['validation_auc'] for record in data]
            
            print(f"Iterations: {iterations}")
            print(f"AUC scores: {auc_scores}")
            print(f"Number of iterations: {len(iterations)}")
            print(f"Number of AUC scores: {len(auc_scores)}")

if __name__ == "__main__":
    test_load_data() 