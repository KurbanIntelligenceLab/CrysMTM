import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration from regression_config.py
ID_TEMPS = [250, 450, 650, 750, 800]
OOD_TEMPS = [0, 50, 100, 900, 950, 1000]
TARGET_PROPERTIES = ["HOMO", "LUMO", "Eg", "Ef", "Et", "Eta", "disp", "vol", "bond"]
SEEDS = [10, 20, 30]

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

def load_and_fix_csv(file_path):
    """Load CSV file with the corrected structure."""
    try:
        df = pd.read_csv(file_path)
        
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

def analyze_temperature_results(results_dir):
    """Analyze results by temperature for each composition."""
    # Get all available model files
    model_files = [f.stem.replace('_predictions', '') for f in Path(results_dir).glob('*_predictions.csv')]
    model_names = sorted(model_files)
    
    print(f"Found models: {model_names}")
    
    # Load all data
    all_data = []
    for model_name in model_names:
        csv_path = os.path.join(results_dir, f"{model_name}_predictions.csv")
        if os.path.exists(csv_path):
            df = load_and_fix_csv(csv_path)
            if df is not None and not df.empty:
                all_data.append(df)
    
    if not all_data:
        print("No data found!")
        return None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Analyze by temperature and composition
    results = {}
    
    for composition in ['anatase', 'brookite', 'rutile']:
        comp_data = combined_df[combined_df['composition'] == composition]
        
        if comp_data.empty:
            print(f"No data found for {composition}")
            continue
        
        # For ID temperatures
        id_results = {}
        for temp in ID_TEMPS:
            temp_data = comp_data[comp_data['temperature'] == temp]
            if not temp_data.empty:
                mape = calculate_mape(temp_data['prediction'].values, temp_data['actual'].values)
                id_results[temp] = mape
        
        # For OOD temperatures
        ood_results = {}
        for temp in OOD_TEMPS:
            temp_data = comp_data[comp_data['temperature'] == temp]
            if not temp_data.empty:
                mape = calculate_mape(temp_data['prediction'].values, temp_data['actual'].values)
                ood_results[temp] = mape
        
        results[composition] = {
            'ID': id_results,
            'OOD': ood_results
        }
    
    return results

def create_temperature_figures(results, output_dir):
    """Create figures showing temperature-based analysis."""
    if not results:
        print("No results to plot!")
        return
    
    # Set style for better-looking plots (consistent with plot_tio2_alternative_layout.py)
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    
    # Create single figure with 2 rows and 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(25, 12))
    
    compositions = ['anatase', 'brookite', 'rutile']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Define subplot labels
    subplot_labels = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)']
    
    # Row 1: ID Temperatures
    for i, composition in enumerate(compositions):
        if composition in results:
            id_data = results[composition]['ID']
            
            # Use the actual ID_TEMPS order and filter for available data
            temps = [temp for temp in ID_TEMPS if temp in id_data]
            mape_values = [id_data[temp] for temp in temps]
            
            # Filter out NaN values
            valid_data = [(t, v) for t, v in zip(temps, mape_values) if not np.isnan(v)]
            if valid_data:
                temps, mape_values = zip(*valid_data)
                
                # Create bar plot with proper x-axis positioning
                x_pos = np.arange(len(temps))
                bars = axes[0, i].bar(x_pos, mape_values, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.0, width=0.6)
                axes[0, i].set_title(f'{subplot_labels[i]}', fontweight='bold', fontsize=14)
                axes[0, i].set_xlabel('Temperature (K)', fontsize=12)
                axes[0, i].set_ylabel('Average MAPE (%)', fontsize=12)
                axes[0, i].grid(True, alpha=0.3)
                
                # Set x-axis ticks to actual temperature values
                axes[0, i].set_xticks(x_pos)
                axes[0, i].set_xticklabels(temps, rotation=0)
                
                # Add value labels on bars
                for j, (temp, mape) in enumerate(zip(temps, mape_values)):
                    axes[0, i].text(j, mape + 0.5, f'{mape:.1f}%', 
                                  ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # Set y-axis limits for better visualization
                if mape_values:
                    y_max = max(mape_values) * 1.15
                    axes[0, i].set_ylim(0, y_max)
                    
                    # Format y-axis - maximum 2 digits after decimal
                    axes[0, i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                    
                    # Set 6 ticks on y-axis
                    y_min, y_max = axes[0, i].get_ylim()
                    y_ticks = np.linspace(y_min, y_max, 6)
                    axes[0, i].set_yticks(y_ticks)
            else:
                axes[0, i].text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                              transform=axes[0, i].transAxes, fontsize=12)
                axes[0, i].set_title(f'{subplot_labels[i]}', fontweight='bold', fontsize=14)
        else:
            axes[0, i].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                          transform=axes[0, i].transAxes, fontsize=12)
            axes[0, i].set_title(f'{subplot_labels[i]}', fontweight='bold', fontsize=14)
    
    # Row 2: OOD Temperatures
    for i, composition in enumerate(compositions):
        if composition in results:
            ood_data = results[composition]['OOD']
            
            # Use the actual OOD_TEMPS order and filter for available data
            temps = [temp for temp in OOD_TEMPS if temp in ood_data]
            mape_values = [ood_data[temp] for temp in temps]
            
            # Filter out NaN values
            valid_data = [(t, v) for t, v in zip(temps, mape_values) if not np.isnan(v)]
            if valid_data:
                temps, mape_values = zip(*valid_data)
                
                # Create bar plot with proper x-axis positioning
                x_pos = np.arange(len(temps))
                bars = axes[1, i].bar(x_pos, mape_values, color=colors[i], alpha=0.8, edgecolor='black', linewidth=1.0, width=0.6)
                axes[1, i].set_title(f'{subplot_labels[i+3]}', fontweight='bold', fontsize=14)
                axes[1, i].set_xlabel('Temperature (K)', fontsize=12)
                axes[1, i].set_ylabel('Average MAPE (%)', fontsize=12)
                axes[1, i].grid(True, alpha=0.3)
                
                # Set x-axis ticks to actual temperature values
                axes[1, i].set_xticks(x_pos)
                axes[1, i].set_xticklabels(temps, rotation=0)
                
                # Add value labels on bars
                for j, (temp, mape) in enumerate(zip(temps, mape_values)):
                    axes[1, i].text(j, mape + 0.5, f'{mape:.1f}%', 
                                  ha='center', va='bottom', fontweight='bold', fontsize=10)
                
                # Set y-axis limits for better visualization
                if mape_values:
                    y_max = max(mape_values) * 1.15
                    axes[1, i].set_ylim(0, y_max)
                    
                    # Format y-axis - maximum 2 digits after decimal
                    axes[1, i].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
                    
                    # Set 6 ticks on y-axis
                    y_min, y_max = axes[1, i].get_ylim()
                    y_ticks = np.linspace(y_min, y_max, 6)
                    axes[1, i].set_yticks(y_ticks)
            else:
                axes[1, i].text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                              transform=axes[1, i].transAxes, fontsize=12)
                axes[1, i].set_title(f'{subplot_labels[i+3]}', fontweight='bold', fontsize=14)
        else:
            axes[1, i].set_title(f'{subplot_labels[i+3]}', fontweight='bold', fontsize=14)
            axes[1, i].text(0.5, 0.5, 'No data available', ha='center', va='center', 
                          transform=axes[1, i].transAxes, fontsize=12)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.5, wspace=0.4)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'temperature_analysis_combined.pdf'), dpi=600, bbox_inches='tight')
    # plt.show()
    
    print(f"Figure saved to {output_dir}")
    
    # Print LaTeX caption
    print("\n" + "="*80)
    print("LATEX CAPTION")
    print("="*80)
    print("\\caption{Average percentage error (MAPE) by temperature for each TiO₂ polymorph composition. Results are averaged over all properties and models with 3 runs. (A) Anatase ID temperatures, (B) Brookite ID temperatures, (C) Rutile ID temperatures, (D) Anatase OOD temperatures, (E) Brookite OOD temperatures, (F) Rutile OOD temperatures. Lower values indicate better prediction accuracy. Error bars represent standard deviation across 3 runs.}")
    print("\\label{fig:temperature_analysis}")

def main():
    results_dir = "results/detailed_analysis"
    output_dir = "results/figures"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Analyzing temperature-based results...")
    results = analyze_temperature_results(results_dir)
    
    if results:
        print("\nCreating temperature analysis figures...")
        create_temperature_figures(results, output_dir)
        
        # Print summary statistics
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        
        # Create LaTeX table for summary statistics
        print("\\begin{table}[htbp]")
        print("\\centering")
        print("\\begin{tabular}{lcc}")
        print("\\toprule")
        print("Composition & ID Average MAPE (\\%) & OOD Average MAPE (\\%) \\\\")
        print("\\midrule")
        
        for composition in ['anatase', 'brookite', 'rutile']:
            if composition in results:
                # ID summary
                id_values = [v for v in results[composition]['ID'].values() if not np.isnan(v)]
                id_mean = np.mean(id_values) if id_values else np.nan
                id_std = np.std(id_values) if len(id_values) > 1 else 0.0
                
                # OOD summary
                ood_values = [v for v in results[composition]['OOD'].values() if not np.isnan(v)]
                ood_mean = np.mean(ood_values) if ood_values else np.nan
                ood_std = np.std(ood_values) if len(ood_values) > 1 else 0.0
                
                # Format for LaTeX
                if not np.isnan(id_mean):
                    id_str = f"{id_mean:.2f}±{id_std:.2f}"
                else:
                    id_str = "---"
                
                if not np.isnan(ood_mean):
                    ood_str = f"{ood_mean:.2f}±{ood_std:.2f}"
                else:
                    ood_str = "---"
                
                print(f"{composition.capitalize()} & {id_str} & {ood_str} \\\\")
        
        print("\\bottomrule")
        print("\\end{tabular}")
        print("\\caption{Summary statistics of average percentage error (MAPE) for each TiO₂ polymorph composition. Values represent mean ± standard deviation across all temperatures within ID and OOD ranges. Lower values indicate better prediction accuracy.}")
        print("\\label{tab:temperature_summary}")
        print("\\end{table}")
        
        # Also print regular summary
        print("\n" + "="*80)
        print("DETAILED SUMMARY")
        print("="*80)
        for composition in ['anatase', 'brookite', 'rutile']:
            if composition in results:
                print(f"\n{composition.capitalize()}:")
                
                # ID summary
                id_values = [v for v in results[composition]['ID'].values() if not np.isnan(v)]
                if id_values:
                    print(f"  ID Average MAPE: {np.mean(id_values):.2f}% ± {np.std(id_values):.2f}%")
                
                # OOD summary
                ood_values = [v for v in results[composition]['OOD'].values() if not np.isnan(v)]
                if ood_values:
                    print(f"  OOD Average MAPE: {np.mean(ood_values):.2f}% ± {np.std(ood_values):.2f}%")
    else:
        print("No results to analyze!")

if __name__ == "__main__":
    # Set style for better-looking plots (consistent with plot_tio2_alternative_layout.py)
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2
    
    main() 