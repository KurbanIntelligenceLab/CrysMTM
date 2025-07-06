import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration from regression_config.py
ID_TEMPS = [250, 450, 650, 750, 800]
OOD_TEMPS = [0, 50, 100, 900, 950, 1000]
TARGET_PROPERTIES = ["HOMO", "LUMO", "Eg", "Ef", "Et", "Eta", "disp", "vol", "bond"]

# Property name mappings for LaTeX
PROPERTY_LATEX = {
    "HOMO": "$E_H$",
    "LUMO": "$E_L$", 
    "Eg": "$E_g$",
    "Ef": "$E_f$",
    "Et": "$E_T$",
    "Eta": "$E_{Ta}$",
    "disp": "$\\Delta r_{max}$",
    "vol": "$\\Delta V$",
    "bond": "$\\Delta d_{Ti-O}$"
}

# Model name mappings for better display
MODEL_NAMES = {
    "egnn": "EGNN",
    "schnet": "SchNet", 
    "pure2dopenet": "Pure2DopeNet",
    "equiformer": "Equiformer",
    "faenet": "FAENet",
    "clip": "CLIP",
    "resnet": "ResNet",
    "vit": "ViT",
    "visnet": "ViSNet",
    "gotennet": "GotenNet"
}

SEEDS = [10, 20, 30]

def load_and_fix_csv(file_path):
    """Load CSV file with the corrected structure."""
    try:
        df = pd.read_csv(file_path)
        
        # The CSV now has the correct structure: model, property, seed, prediction, actual, temperature, composition
        # where 'property' contains the property names and 'actual' contains the target values
        
        # Ensure we have the expected columns
        expected_cols = ['model', 'property', 'seed', 'prediction', 'actual', 'temperature', 'composition']
        if not all(col in df.columns for col in expected_cols):
            print(f"Warning: Missing expected columns in {file_path}")
            print(f"Available columns: {list(df.columns)}")
            return None
        
        print(f"Successfully loaded {file_path} with {len(df)} samples")
        return df
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def calculate_mape(predictions, actuals):
    """Calculate Mean Average Percentage Error."""
    if len(predictions) == 0 or len(actuals) == 0:
        return np.nan
    
    # Filter out any invalid values
    valid_mask = np.isfinite(predictions) & np.isfinite(actuals) & (actuals != 0)
    
    if not np.any(valid_mask):
        return np.nan
    
    predictions = np.array(predictions)[valid_mask]
    actuals = np.array(actuals)[valid_mask]
    
    # Calculate MAPE
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    return mape

def analyze_model_results(model_name, results_dir):
    """Analyze results for a specific model."""
    csv_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
    
    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return None
    
    print(f"Loading {csv_path}...")
    df = load_and_fix_csv(csv_path)
    
    if df is None or df.empty:
        print(f"Warning: No data found in {csv_path}")
        return None
    
    # Ensure we have the expected columns
    expected_cols = ['model', 'property', 'seed', 'prediction', 'actual', 'temperature', 'composition']
    if not all(col in df.columns for col in expected_cols):
        print(f"Warning: Missing expected columns in {csv_path}")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    results = {}
    
    # For each property
    for property_name in TARGET_PROPERTIES:
        if property_name not in df['property'].values:
            print(f"Warning: Property {property_name} not found in {model_name} results")
            continue
            
        property_data = df[df['property'] == property_name]
        
        # For each seed
        seed_results = {}
        for seed in SEEDS:
            if seed not in property_data['seed'].values:
                print(f"Warning: Seed {seed} not found for {model_name}/{property_name}")
                continue
                
            seed_data = property_data[property_data['seed'] == seed]
            
            # Separate ID and OOD temperatures
            id_data = seed_data[seed_data['temperature'].isin(ID_TEMPS)]
            ood_data = seed_data[seed_data['temperature'].isin(OOD_TEMPS)]
            
            # Calculate MAPE for ID and OOD
            id_mape = calculate_mape(id_data['prediction'].values, id_data['actual'].values)
            ood_mape = calculate_mape(ood_data['prediction'].values, ood_data['actual'].values)
            
            seed_results[seed] = {
                'ID': id_mape,
                'OOD': ood_mape
            }
        
        # Average and std over seeds
        if seed_results:
            id_values = [result['ID'] for result in seed_results.values() if not np.isnan(result['ID'])]
            ood_values = [result['OOD'] for result in seed_results.values() if not np.isnan(result['OOD'])]
            
            avg_id_mape = np.nanmean(id_values) if id_values else np.nan
            avg_ood_mape = np.nanmean(ood_values) if ood_values else np.nan
            std_id_mape = np.nanstd(id_values) if len(id_values) > 1 else 0.0
            std_ood_mape = np.nanstd(ood_values) if len(ood_values) > 1 else 0.0
            
            results[property_name] = {
                'ID': avg_id_mape,
                'OOD': avg_ood_mape,
                'ID_std': std_id_mape,
                'OOD_std': std_ood_mape
            }
    
    return results

def print_latex_table(all_results):
    """Print results as a LaTeX table."""
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{l" + "cc" * len(TARGET_PROPERTIES) + "}")
    print("\\toprule")
    
    # Header row 1: Property names
    header1 = "\\multirow{2}{*}{Model}"
    for prop in TARGET_PROPERTIES:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{PROPERTY_LATEX[prop]}}}"
    header1 += " \\\\"
    print(header1)
    
    # Header row 2: ID/OOD
    header2 = ""
    for prop in TARGET_PROPERTIES:
        header2 += " & ID & OOD"
    header2 += " \\\\"
    print(header2)
    
    print("\\midrule")
    
    # Find best and second best for each property/type
    best_scores = {}
    for prop in TARGET_PROPERTIES:
        best_scores[prop] = {'ID': [], 'OOD': []}
        for model_name in all_results:
            if prop in all_results[model_name]:
                id_val = all_results[model_name][prop]['ID']
                ood_val = all_results[model_name][prop]['OOD']
                if not np.isnan(id_val):
                    best_scores[prop]['ID'].append((id_val, model_name))
                if not np.isnan(ood_val):
                    best_scores[prop]['OOD'].append((ood_val, model_name))
        
        # Sort and get best/second best
        best_scores[prop]['ID'].sort(key=lambda x: x[0])
        best_scores[prop]['OOD'].sort(key=lambda x: x[0])
    
    # Data rows
    for model_name in sorted(all_results.keys()):
        display_name = MODEL_NAMES.get(model_name, model_name)
        row = f"{display_name}"
        
        for prop in TARGET_PROPERTIES:
            if prop in all_results[model_name]:
                id_val = all_results[model_name][prop]['ID']
                ood_val = all_results[model_name][prop]['OOD']
                id_std = all_results[model_name][prop]['ID_std']
                ood_std = all_results[model_name][prop]['OOD_std']
                
                # Format values with 2 decimal places and std
                if not np.isnan(id_val):
                    id_str = f"{id_val:.2f}±{id_std:.2f}"
                    # Check if best or second best
                    if best_scores[prop]['ID'] and best_scores[prop]['ID'][0][1] == model_name:
                        id_str = f"\\textbf{{{id_str}}}"
                    elif len(best_scores[prop]['ID']) > 1 and best_scores[prop]['ID'][1][1] == model_name:
                        id_str = f"\\underline{{{id_str}}}"
                else:
                    id_str = "---"
                
                if not np.isnan(ood_val):
                    ood_str = f"{ood_val:.2f}±{ood_std:.2f}"
                    # Check if best or second best
                    if best_scores[prop]['OOD'] and best_scores[prop]['OOD'][0][1] == model_name:
                        ood_str = f"\\textbf{{{ood_str}}}"
                    elif len(best_scores[prop]['OOD']) > 1 and best_scores[prop]['OOD'][1][1] == model_name:
                        ood_str = f"\\underline{{{ood_str}}}"
                else:
                    ood_str = "---"
                
                row += f" & {id_str} & {ood_str}"
            else:
                row += " & --- & ---"
        
        row += " \\\\"
        print(row)
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("}")
    print("\\caption{Performance comparison of different deep learning models on material property prediction tasks. Results show Mean Absolute Percentage Error (MAPE) averaged over 3 independent runs with different random seeds. Values represent mean ± standard deviation. MAPE is used to ensure fair comparison across properties with different scales and units. Best performing model for each property and evaluation setting is highlighted in \\textbf{bold}, while second best is \\underline{underlined}. Lower values indicate better prediction accuracy.}")
    print("\\label{tab:model_comparison}")
    print("\\end{table}")

def main():
    results_dir = "results/detailed_analysis"
    
    # Get all available model files
    model_files = [f.stem.replace('_predictions', '') for f in Path(results_dir).glob('*_predictions.csv')]
    model_names = sorted(model_files)
    
    print(f"Found models: {model_names}")
    
    # Analyze each model
    all_results = {}
    for model_name in model_names:
        print(f"\nAnalyzing {model_name}...")
        results = analyze_model_results(model_name, results_dir)
        if results:
            all_results[model_name] = results
    
    # Print LaTeX table
    if not all_results:
        print("No results to analyze!")
        return
    
    print("\n" + "="*80)
    print("LATEX TABLE")
    print("="*80)
    print_latex_table(all_results)

if __name__ == "__main__":
    main() 