import argparse
import numpy as np
import pandas as pd
import json
import os
import glob
from itertools import product
import subprocess

# Define parameter options
PARAMS_ABLATION = {
    'dataset': ["SMAP_", "SMD_", "MSL_", "NIPS_TS_Swan_", "NIPS_TS_creditcard_", "NIPS_TS_Water_", "UCR_1", "UCR_2"],
    'model': ['COUTA'],
    'epochs': ['50'],
    'seq_len': ['50'],
    'batch_size': ['64'],
    'sigma': ['0.1', '0.5', '1.0'],
    'w': ['4'],
    'smooth_count': ['500'],
    'seed': ['0']
}

PARAMS_INIT = {
    'dataset': ["SMAP_", "SMD_", "MSL_", "NIPS_TS_Swan_", "NIPS_TS_creditcard_", "NIPS_TS_Water_", "UCR_1", "UCR_2"],
    'model': ["COUTA", "TimesNet", "DeepSVDDTS"],
    'epochs': ['50'],
    'seq_len': ['50'],
    'batch_size': ['64'],
    'sigma': ['0.5'],
    'w': ['4'],
    'smooth_count': ['500'],
    'seed': ['0']
}

PARAMS_ATTACK = {
    'dataset': ['MSL_', 'SMD_', "NIPS_TS_Water_", "UCR_1"],
    'model': ['COUTA'],
    'epochs': ['100'],
    'seq_len': ['50'],
    'batch_size': ['64'],
    'sigma': ['0.5', '1.0'],
    'w': ['4'],
    'smooth_count': ['500'],
    'seed': ['0']
}


PARAMS = PARAMS_ABLATION

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Anomaly Detection')
    parser.add_argument('--exp_folder', type=str, default='exps/init', help='Experiment folder')
    parser.add_argument('--output', type=str, default=None, 
                        help='Path to save collected results (default: {exp_folder}/collected_results.json)')
    parser.add_argument('--rerun_results', type=bool, default=False, help='Re-run results.json files')
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

def metrics_parser(metrics, header):
    # Convert metrics to a dictionary
    metrics_dict = {
        f'{header} ROC AUC': metrics[0],
        f'{header} Average Precision': metrics[1],
        f'{header} F1': metrics[2],
        f'{header} Precision': metrics[3],
        f'{header} Recall': metrics[4],
    }
    return metrics_dict

def radii_stats_parser(radii_stats, header):
    radii_stats_dict = {
        f'{header} Mean': radii_stats['mean'],
        f'{header} Std': radii_stats['std'],
        # f'{header} Min': radii_stats['min'],
        f'{header} Max': radii_stats['max'],
        f'{header} Proportion': radii_stats['proportion'],
    }
    return radii_stats_dict

def certified_stats_parser(certified_stats, radii_thresholds, header):
    certified_acc = certified_stats['accuracy']
    certified_f1 = certified_stats['f1']

    certified_dict = {
        f'{header} Accuracy': certified_acc,
        f'{header} F1': certified_f1,
        f'{header} Radii Thresholds': radii_thresholds,
    }
    return certified_dict



def process_results(params, results_data):
    dataset, model, epochs, seq_len, batch_size, sigma, w, smooth_count, seed = params
    param_dict = {
        'Dataset': dataset.strip("_").replace("_", " ").upper(),
        'Model': model,
        'Epochs': epochs,
        'Sequence Length': seq_len,
        'Batch Size': batch_size,
        'Sigma': sigma,
        'Window Size': w,
        'Smooth Count': smooth_count,
        'Seed': seed
    }
    param_names = param_dict.keys()

    metric_dict = {}
    clean_metrics = results_data['clean_metrics']
    clean_adj_metrics = results_data['clean_adj_metrics']
    smoothed_metrics = results_data['smoothed_metrics'] 
    smoothed_adj_metrics = results_data['smoothed_adj_metrics']
    radii_stats = results_data['certified_stats']['radii_stats']
    radii_adj_stats = results_data['certified_adj_stats']['radii_stats']
    
    try:
        attack_clean_metrics = results_data['adv_clean_metrics']
        attack_smoothed_metrics = results_data['adv_smoothed_metrics']
    except KeyError:
        pass
    
    certified_eva = results_data['certified_adj_stats']['evasion']
    certified_ava = results_data['certified_adj_stats']['availability']
    radii_thresholds = results_data['radii_thresholds']


    metric_dict.update(metrics_parser(clean_metrics, 'Normal without adj'))
    metric_dict.update(metrics_parser(clean_adj_metrics, 'Normal'))
    metric_dict.update(metrics_parser(smoothed_metrics, 'Smoothed without adj'))
    metric_dict.update(metrics_parser(smoothed_adj_metrics, 'Smoothed'))
    metric_dict.update(radii_stats_parser(radii_stats, 'Radii without adj'))
    metric_dict.update(radii_stats_parser(radii_adj_stats, 'Radii'))
    try:
        metric_dict.update(metrics_parser(attack_clean_metrics, 'Attacked'))
        metric_dict.update(metrics_parser(attack_smoothed_metrics, 'Attacked Smoothed'))
    except:
        pass
    metric_dict.update(certified_stats_parser(certified_eva, radii_thresholds, 'Certified Evasion'))
    metric_dict.update(certified_stats_parser(certified_ava, radii_thresholds, 'Certified Availability'))

    metric_names = metric_dict.keys()

    results_summary= {}
    results_summary.update(param_dict)
    results_summary.update(metric_dict)

    return results_summary, param_names, metric_names

def save_csv(all_results, param_names, metric_names):
    # Reorder columns to put parameters first, excluding Folder
    results_df = pd.DataFrame(all_results)
    results_df = results_df[list(param_names) + list(metric_names)]
    # Format float columns
    for col in metric_names:
        results_df[col] = results_df[col].astype(float).map(lambda x: f"{x:.5f}")
    # Save to CSV
    csv_output_path = args.output if args.output else os.path.join(args.exp_folder, "collected_results.csv")
    results_df.to_csv(csv_output_path, index=False)
    print(f"\nSaved results to CSV: {csv_output_path}")


def plot_results(all_results, param_names, metric_names):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker  # Added import for tick formatting
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Create output directory for plots
    plots_dir = os.path.join(args.exp_folder, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get unique datasets and sigmas
    datasets = results_df['Dataset'].unique()
    sigmas = sorted(results_df['Sigma'].unique(), key=float)  # Sort sigmas numerically
    
    # Define attack types and metrics
    attack_types = ['Certified Evasion', 'Certified Availability']
    metric_types = ['F1', 'Accuracy']
    
    # For each dataset
    for dataset in datasets:
        dataset_df = results_df[results_df['Dataset'] == dataset]
        
        # Create a 2x2 grid of subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))  # Reduced height
        plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Reduced vertical spacing
        
        # Flatten for easier indexing
        axes = axes.flatten()
        
        # Counter for subplot position
        subplot_idx = 0
        
        # For each attack type and metric combination
        for attack_type in attack_types:
            for metric_type in metric_types:
                metric_key = f'{attack_type} {metric_type}'
                
                if metric_key not in metric_names:
                    subplot_idx += 1
                    continue
                
                ax = axes[subplot_idx]
                
                # For each sigma (noise level)
                legend_entries = []
                for sigma in sigmas:
                    sigma_df = dataset_df[dataset_df['Sigma'] == sigma]
                    
                    if len(sigma_df) == 0:
                        continue
                    
                    # Process each row for this sigma
                    for _, row in sigma_df.iterrows():
                        # Get the radii thresholds and metric values
                        thresholds = row[f'{attack_type} Radii Thresholds']
                        metric_values = row[metric_key]
                        
                        # Check if data exists and is in expected format
                        if isinstance(thresholds, list) and isinstance(metric_values, list):
                            # Plot the line
                            ax.plot(thresholds, metric_values, label=f'σ={sigma}', linewidth=4)
                            if f'σ={sigma}' not in legend_entries:
                                legend_entries.append(f'σ={sigma}')
                
                fontsize = 32

                title_convert = {
                    'Certified Evasion': 'Evasion Attack',
                    'Certified Availability': 'Availability Attack',
                }

                ax.set_title(f'{title_convert[attack_type]}', fontsize=fontsize)
                
                ax.set_xlabel('Radius Threshold', fontsize=fontsize)
                
                ax.set_ylabel(f'Certified {metric_type}', fontsize=fontsize)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='both', which='major', labelsize=fontsize)
                
                # Set y-axis ticks to show only 1 decimal place
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
                
                if legend_entries:
                    ax.legend(fontsize=fontsize)
                
                subplot_idx += 1
        
        # Save the plot
        clean_dataset_name = dataset.replace(' ', '_')
        plot_name = f"{clean_dataset_name}_certification_plots.png"
        plt.savefig(os.path.join(plots_dir, plot_name), bbox_inches='tight', dpi=300)
        plt.close()
    
    print(f"Plots saved in {plots_dir}")


if __name__ == '__main__':
    args = parse_args()
   
    print(f"Searching in folder: {args.exp_folder}")
    
    # Find all matching folders
    matching_folders, all_combinations = find_matching_folders(args.exp_folder)
    
    # Print results
    total_combinations = len(all_combinations)
    found_combinations = sum(1 for folders in matching_folders.values() if folders)
    
    print(f"\nFound: {found_combinations}/{total_combinations} expected results")

    if found_combinations != total_combinations:
        print("Warning: Not all expected results were found.")
    # Print the unfound combinations
    unfound_combinations = [params for params, folders in matching_folders.items() if not folders]
    if unfound_combinations:
        print("Unfound combinations:")
        for params in unfound_combinations:
            print(f"dataset: {params[0]}, model: {params[1]}, epochs: {params[2]}, seq_len: {params[3]}, batch_size: {params[4]}, sigma: {params[5]}, w: {params[6]}, smooth_count: {params[7]}, seed: {params[8]}")
        exit(0)
    
    # Initialize list to store collected results for CSV
    all_results = []
        
    for params, folders in matching_folders.items():
        folder = folders[0]
        if len(folders) > 1:
            print(f"Warning: Multiple folders found for {params}, using the first one")
        if not os.path.exists(os.path.join(folder, 'results.json')):
            print(f"Warning: No results.json found in {folder}")
            continue
        # Re-run the results.json files
        if args.rerun_results:
            print("Re-running results.json files")
            print(f'Running {params}')

            # read the results.json file to get the arguments
            with open(os.path.join(folder, 'results.json'), 'r') as f:
                exp_args = json.load(f)['args']
            # change the arguments
            exp_args['exps_root'] = args.exp_folder
            exp_args['load_model'] = os.path.join(folder, 'model.pkl')
            exp_args['load_noise_scores'] = os.path.join(folder, 'saved_noise_scores.pkl')
            # run the experiment
            cmd = ["python", "run.py"]
            for key, value in exp_args.items():
                cmd.extend([f"--{key}", str(value)])
                
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd) 

        # Read results.json file and process results
        with open(os.path.join(folder, 'results.json'), 'r') as f:
            results_data = json.load(f)

        results_summary, param_names, metric_names = process_results(params, results_data)
        all_results.append(results_summary)
        
    # Save to CSV
    # param_names = list(param_names)
    # param_names = ['Dataset', 'Model', 'Sigma']
    # metric_names = ['Normal F1', 'Normal ROC AUC', 'Smoothed F1', 'Smoothed ROC AUC', 'Radii Mean', 'Radii Max', 'Radii Std', 'Radii Proportion']
    # metric_names = ['Normal F1', 'Normal ROC AUC', 'Attacked F1', 'Attacked ROC AUC', 'Attacked Smoothed F1', 'Attacked Smoothed ROC AUC']
    # save_csv(all_results, param_names, metric_names)

    # Plot results
    plot_results(all_results, param_names, metric_names)




