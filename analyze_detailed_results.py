import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration from regression_config.py
ID_TEMPS = [250, 450, 650, 750, 800]
OOD_TEMPS = [0, 50, 100, 900, 950, 1000]
TARGET_PROPERTIES = ["HOMO", "LUMO", "Eg", "Ef", "Et", "Eta", "disp", "vol", "bond"]
SEEDS = [10, 20, 30]

def load_and_fix_csv(file_path):
    """Load CSV file and handle the column structure issue."""
    try:
        df = pd.read_csv(file_path)
        
        # The CSV has the structure: model, target, seed, prediction, temperature, composition
        # where 'target' contains the actual target values (not property names)
        # The property names were lost due to duplicate column names in the evaluation script
        
        # Extract model name from file path
        model_name = os.path.basename(file_path).replace('_predictions.csv', '')
        
        # Rename the target column to actual
        if 'target' in df.columns:
            df = df.rename(columns={'target': 'actual'})
        
        # Add model column if not present
        if 'model' not in df.columns:
            df['model'] = model_name
        
        # Since the evaluation script loops through TARGET_PROPERTIES and SEEDS,
        # and appends results for each combination, we can infer the property names
        # by grouping the data appropriately
        
        # The data is organized as:
        # - First N rows: HOMO for seed 10
        # - Next N rows: HOMO for seed 20
        # - Next N rows: HOMO for seed 30
        # - Next N rows: LUMO for seed 10
        # - Next N rows: LUMO for seed 20
        # - Next N rows: LUMO for seed 30
        # - And so on for all properties...
        
        # Calculate samples per property per seed
        unique_seeds = sorted(df['seed'].unique())
        if unique_seeds:
            samples_per_seed = len(df[df['seed'] == unique_seeds[0]])
            samples_per_property_per_seed = samples_per_seed // len(unique_seeds)
            
            print(f"Detected {samples_per_seed} samples per seed, {samples_per_property_per_seed} samples per property per seed")
            
            # Split the data into property groups
            property_groups = []
            
            for i, prop in enumerate(TARGET_PROPERTIES):
                # For each property, get all samples across all seeds
                start_idx = i * len(unique_seeds) * samples_per_property_per_seed
                end_idx = (i + 1) * len(unique_seeds) * samples_per_property_per_seed
                
                if end_idx <= len(df):
                    prop_data = df.iloc[start_idx:end_idx].copy()
                    prop_data['property'] = prop
                    property_groups.append(prop_data)
                else:
                    # Handle the case where we don't have complete data for all properties
                    print(f"Warning: Incomplete data for property {prop}")
                    break
            
            if property_groups:
                df = pd.concat(property_groups, ignore_index=True)
            else:
                # Fallback: assign 'unknown' to all
                df['property'] = 'unknown'
        else:
            df['property'] = 'unknown'
        
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
        
        # Average over seeds
        if seed_results:
            avg_id_mape = np.nanmean([result['ID'] for result in seed_results.values()])
            avg_ood_mape = np.nanmean([result['OOD'] for result in seed_results.values()])
            
            results[property_name] = {
                'ID': avg_id_mape,
                'OOD': avg_ood_mape
            }
    
    return results

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
    
    # Create summary DataFrame
    if not all_results:
        print("No results to analyze!")
        return
    
    # Create a multi-level column DataFrame
    columns = []
    for prop in TARGET_PROPERTIES:
        columns.extend([(prop, 'ID'), (prop, 'OOD')])
    
    summary_data = []
    for model_name in all_results:
        row_data = []
        for prop in TARGET_PROPERTIES:
            if prop in all_results[model_name]:
                row_data.extend([
                    all_results[model_name][prop]['ID'],
                    all_results[model_name][prop]['OOD']
                ])
            else:
                row_data.extend([np.nan, np.nan])
        summary_data.append(row_data)
    
    # Create DataFrame with multi-level columns
    summary_df = pd.DataFrame(summary_data, index=all_results.keys(), columns=pd.MultiIndex.from_tuples(columns))
    
    # Save results
    output_path = os.path.join(results_dir, "model_comparison_mape.csv")
    summary_df.to_csv(output_path)
    
    print(f"\nResults saved to: {output_path}")
    print("\nSummary:")
    print(summary_df.round(2))
    
    # Also create a simplified version with just the mean across all properties
    simplified_data = []
    for model_name in all_results:
        id_mape_values = [all_results[model_name][prop]['ID'] for prop in TARGET_PROPERTIES if prop in all_results[model_name]]
        ood_mape_values = [all_results[model_name][prop]['OOD'] for prop in TARGET_PROPERTIES if prop in all_results[model_name]]
        
        avg_id = np.nanmean(id_mape_values) if id_mape_values else np.nan
        avg_ood = np.nanmean(ood_mape_values) if ood_mape_values else np.nan
        
        simplified_data.append({
            'Model': model_name,
            'Avg_ID_MAPE': avg_id,
            'Avg_OOD_MAPE': avg_ood
        })
    
    simplified_df = pd.DataFrame(simplified_data)
    simplified_path = os.path.join(results_dir, "model_comparison_simplified.csv")
    simplified_df.to_csv(simplified_path, index=False)
    
    print(f"\nSimplified results saved to: {simplified_path}")
    print("\nSimplified Summary (Average across all properties):")
    print(simplified_df.round(2))

if __name__ == "__main__":
    main() 