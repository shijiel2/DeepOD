import argparse
import numpy as np
import pandas as pd
import json
import os
from deepod.metrics import ts_metrics, point_adjustment
import glob
from itertools import product


# Define parameter options
PARAMS = {
    'dataset': ['UCR_1', 'UCR_2', 'UCR_3'],
    'model': ['TimesNet', 'COUTA', 'DeepSVDDTS'],
    'epochs': ['20'],
    'seq_len': ['10'],
    'batch_size': ['64'],
    'sigma': ['0.1', '0.3', '0.5'],
    'w': ['2', '4'],
    'smooth_count': ['500'],
    'seed': ['0']
}

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Anomaly Detection')
    parser.add_argument('--exp_folder', type=str, default='exps/deeplearn', help='Experiment folder')
    parser.add_argument('--output', type=str, default='exps/deeplearn/collected_results.csv', 
                        help='Path to save collected results (default: {exp_folder}/collected_results.json)')
    return parser.parse_args()

def find_matching_folders(exp_folder):
    results = {}
    
    # Generate all parameter combinations
    param_keys = ['dataset', 'model', 'epochs', 'seq_len', 'batch_size', 'sigma', 'w', 'smooth_count', 'seed']
    param_values = [PARAMS[key] for key in param_keys]
    all_combinations = list(product(*param_values))
    
    # For each combination of parameters
    for values in all_combinations:
        # Create a dict of the current parameters
        param_dict = dict(zip(param_keys, values))
        
        # Create the pattern to match (excluding ID)
        pattern = "{dataset}_{model}_epochs_{epochs}_seq_len_{seq_len}_batch_size_{batch_size}_sigma_{sigma}_w_{w}_smooth_count_{smooth_count}_seed_{seed}".format(**param_dict)
        
        # Look for matching folders
        search_pattern = os.path.join(exp_folder, f"{pattern}_id_*")
        matching_folders = glob.glob(search_pattern)
        
        # Store results for all combinations
        results[tuple(values)] = matching_folders
    
    return results, all_combinations

def process_results(results_data):
    # Convert lists to variables for readability
    clean_metrics = results_data['clean']
    adj_clean_metrics = results_data['clean_adj']
    smoothed_metrics = results_data['smoothed'] 
    adj_smoothed_metrics = results_data['smoothed_adj']
    
    # Extract individual metrics
    clean_roc_auc, clean_average_precision_score, clean_best_f1, clean_best_p, clean_best_r = clean_metrics
    adj_clean_roc_auc, adj_clean_average_precision_score, adj_clean_best_f1, adj_clean_best_p, adj_clean_best_r = adj_clean_metrics
    smoothed_roc_auc, smoothed_average_precision_score, smoothed_best_f1, smoothed_best_p, smoothed_best_r = smoothed_metrics
    adj_smoothed_roc_auc, adj_smoothed_average_precision_score, adj_smoothed_best_f1, adj_smoothed_best_p, adj_smoothed_best_r = adj_smoothed_metrics
    
    # Calculate average radius if available
    try:
        if isinstance(results_data['radiis'], list):
            avg_radii = np.mean(results_data['radiis'])
        else:
            avg_radii = np.mean(results_data['radiis'])
    except (KeyError, TypeError):
        avg_radii = float('nan')
    
    results_summary = {
        'Clean ROC AUC': clean_roc_auc,
        'Clean AP Score': clean_average_precision_score,
        'Clean Best F1': clean_best_f1,
        'Clean Best Precision': clean_best_p,
        'Clean Best Recall': clean_best_r,
        'Adjusted Clean ROC AUC': adj_clean_roc_auc,
        'Adjusted Clean AP Score': adj_clean_average_precision_score,
        'Adjusted Clean Best F1': adj_clean_best_f1,
        'Adjusted Clean Best Precision': adj_clean_best_p,
        'Adjusted Clean Best Recall': adj_clean_best_r,
        'Smoothed ROC AUC': smoothed_roc_auc,
        'Smoothed AP Score': smoothed_average_precision_score,
        'Smoothed Best F1': smoothed_best_f1,
        'Smoothed Best Precision': smoothed_best_p,
        'Smoothed Best Recall': smoothed_best_r,
        'Adjusted Smoothed ROC AUC': adj_smoothed_roc_auc,
        'Adjusted Smoothed AP Score': adj_smoothed_average_precision_score,
        'Adjusted Smoothed Best F1': adj_smoothed_best_f1,
        'Adjusted Smoothed Best Precision': adj_smoothed_best_p,
        'Adjusted Smoothed Best Recall': adj_smoothed_best_r,
        'Average Radius': avg_radii
    }

    return results_summary

if __name__ == '__main__':
    args = parse_args()
   
    print(f"Searching in folder: {args.exp_folder}")
    
    # Find all matching folders
    matching_folders, all_combinations = find_matching_folders(args.exp_folder)
    
    # Print results
    total_combinations = len(all_combinations)
    found_combinations = sum(1 for folders in matching_folders.values() if folders)
    
    print(f"\nFound: {found_combinations}/{total_combinations} expected results")
    
    # Initialize list to store collected results for CSV
    all_results = []
    missing_results_files = []
    
    # Process found folders and collect results
    for params, folders in matching_folders.items():
        if not folders:
            continue
            
        dataset, model, epochs, seq_len, batch_size, sigma, w, smooth_count, seed = params
        
        for folder in folders:
            results_file = os.path.join(folder, "results.json")
            folder_name = os.path.basename(folder)
            
            if not os.path.exists(results_file):
                print(f"Warning: No results.json found in {folder_name}")
                missing_results_files.append(folder_name)
                continue
                
            try:
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    
                    # Process results
                    results_summary = process_results(results_data)
                    
                    # Add parameters to results_summary
                    results_summary.update({
                        'Dataset': dataset,
                        'Model': model,
                        'Epochs': epochs,
                        'Sequence Length': seq_len,
                        'Batch Size': batch_size,
                        'Sigma': sigma,
                        'Window Size': w,
                        'Smooth Count': smooth_count,
                        'Seed': seed,
                        'Folder': folder_name
                    })
                    
                    # Add to collection
                    all_results.append(results_summary)
                    
            except Exception as e:
                print(f"Error loading results from {folder_name}: {str(e)}")
    
    # Print summary
    print(f"\nCollected {len(all_results)} valid results")
    
    if missing_results_files:
        print(f"Missing results.json files in {len(missing_results_files)} folders:")
        for folder in missing_results_files[:5]:
            print(f"  - {folder}")
        if len(missing_results_files) > 5:
            print(f"  ... and {len(missing_results_files) - 5} more")
    
    # Print missing combinations
    if total_combinations - found_combinations > 0:
        print(f"\n=== Missing Experiments ({total_combinations - found_combinations}) ===")
        missing_count = 0
        for params, folders in matching_folders.items():
            if folders:
                continue
            
            dataset, model, epochs, seq_len, batch_size, sigma, w, smooth_count, seed = params
            print(f"Dataset: {dataset}, Model: {model}, σ: {sigma}, w: {w}")
            missing_count += 1
            
            if missing_count >= 10:
                remaining = total_combinations - found_combinations - missing_count
                if remaining > 0:
                    print(f"... and {remaining} more missing experiments")
                break
    
    # Create DataFrame and save as CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Reorder columns to put parameters first, excluding Folder
        param_cols = ['Dataset', 'Model', 'Epochs', 'Sequence Length', 'Batch Size', 'Sigma', 
                      'Window Size', 'Smooth Count', 'Seed']
        metric_cols = [col for col in results_df.columns if col not in param_cols and col != 'Folder']
        results_df = results_df[param_cols + metric_cols]
        
        # Save to CSV
        csv_output_path = args.output if args.output else os.path.join(args.exp_folder, "collected_results.csv")
        results_df.to_csv(csv_output_path, index=False)
        print(f"\nSaved results to CSV: {csv_output_path}")
    else:
        print("\nNo results to save to CSV.")




