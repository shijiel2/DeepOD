import argparse
import numpy as np
import pandas as pd
import json
import os
import logging
import random
import torch
import pickle  # Add pickle import
from deepod.models.time_series import TimesNet, COUTA, DeepSVDDTS, DeepIsolationForestTS, TranAD, SmoothedMedian
from testbed.utils import data_standardize
from deepod.metrics import ts_metrics, point_adjustment

def parse_args():
    parser = argparse.ArgumentParser(description='Time Series Anomaly Detection')
    # General parameters
    parser.add_argument('--data_train', type=str, default='dataset/DCdetector_dataset/UCR/UCR_1_train.npy', help='Train Dataset')
    parser.add_argument('--data_test', type=str, default='dataset/DCdetector_dataset/UCR/UCR_1_test.npy', help='Test Dataset')
    parser.add_argument('--data_test_label', type=str, default='dataset/DCdetector_dataset/UCR/UCR_1_test_label.npy', help='Test Label Dataset')
    parser.add_argument('--exps_root', type=str, default='exps', help='Experiments folder')
    parser.add_argument('--exp_name', type=str, default='test', help='Experiment name')
    parser.add_argument('--subset_size', type=int, default=-1, help='Subset size to use (-1 means all data)')
    parser.add_argument('--model', type=str, default='TimesNet', help='Model to use')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--load_model', type=str, default=None, help='Path to saved model to load (default: None)')
    
    # Common model parameters
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--stride', type=int, default=1, help='Stride')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--epoch_steps', type=int, default=-1, help='Number of steps per epoch (-1 means all data)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    # Model-specific parameters (add when needed)
    
    return parser.parse_args()


def setup_logger(log_path):
    """Set up logger to write to console and file"""
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def set_seed(seed):
    """Set random seed for reproducibility across multiple libraries"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger = logging.getLogger()
    logger.info(f"Random seed set to {seed}")
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    args = parse_args()

    data_train_path = args.data_train
    data_test_path = args.data_test
    data_test_label_path = args.data_test_label
    exp_folder = f'{args.exps_root}/{args.exp_name}'
    os.makedirs(exp_folder, exist_ok=True)
    
    # Setup logging
    log_path = f"{exp_folder}/experiment.log"
    logger = setup_logger(log_path)
    
    # Set random seed globally
    set_seed(args.seed)
    
    logger.info(f"Experiment started: {args.exp_name}")
    logger.info(f"Command line arguments: {vars(args)}")
    
    test_labels = np.load(data_test_label_path)
    X_train_df = pd.DataFrame(np.load(data_train_path))
    X_test_df = pd.DataFrame(np.load(data_test_path))
    X_train, X_test = data_standardize(X_train_df, X_test_df)

    if args.subset_size != -1:
        logger.info(f'Using subset of size {args.subset_size}')
        subset_size = args.subset_size
        X_train = X_train[:subset_size]
        X_test = X_test[:subset_size]
        test_labels = test_labels[:subset_size]
    logger.info(f'Dataset shapes: X_train {X_train.shape}, X_test {X_test.shape}, labels {test_labels.shape}')

    logger.info(f'Load model {args.model}')
    # Common params for all models
    common_params = dict(
        seq_len=args.seq_len,
        stride=args.stride,
        epochs=args.epochs,
        epoch_steps=args.epoch_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    # Set up model with specific parameters
    if args.model == 'TimesNet':
        model_params = {
            **common_params,
            # 'kernel_size': args.kernel_size,
        }
        clf = TimesNet(**model_params)
    elif args.model == 'COUTA':
        model_params = {
            **common_params,
            # 'alpha': args.alpha
        }
        clf = COUTA(**model_params)
    elif args.model == 'DeepSVDDTS':
        model_params = {
            **common_params,
            # 'rep_dim': args.rep_dim
        }
        clf = DeepSVDDTS(**model_params)
    elif args.model == 'DeepIsolationForestTS':
        model_params = {
            **common_params,
            # 'n_trees': args.n_trees
        }
        clf = DeepIsolationForestTS(**model_params)
    elif args.model == 'TranAD':
        model_params = {
            **common_params,
            # 'n_layers': args.n_layers,
        }
        clf = TranAD(**model_params)
    else:
        raise ValueError(f'Invalid model {args.model}')

    # Load saved model if specified
    if args.load_model is not None:
        logger.info(f'Loading model from {args.load_model}')
        try:
            # Load model using pickle instead of PyTorch
            with open(args.load_model, 'rb') as f:
                clf = pickle.load(f)
            logger.info('Model loaded successfully')
        except Exception as e:
            logger.error(f'Failed to load model: {str(e)}')
            exit(1)  # Exit if model loading fails
    else:
        # Train the model
        logger.info('Training start...')
        clf.fit(X_train)

        # Save model using pickle
        model_path = f"{exp_folder}/model.pkl"
        logger.info(f'Saving trained model to {model_path}')
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(clf, f)
            logger.info('Model saved successfully')
        except Exception as e:
            logger.error(f'Failed to save model: {str(e)}')

    # Collect results
    results = {
        'args': vars(args),  # Convert args to dictionary
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    logger.info('Collecting clean results...')
    scores = clf.decision_function(X_test)
    results['clean'] = ts_metrics(test_labels, scores)
    results['clean_adj'] = ts_metrics(test_labels, point_adjustment(test_labels, scores))
    results['clean_scores'] = scores.tolist()

    logger.info('Collecting smoothed results...')
    smoothed_clf = SmoothedMedian(clf, sigma=0.1)
    s_scores, radiis = smoothed_clf.decision_function(X_test)
    results['smoothed'] = ts_metrics(test_labels, s_scores)
    results['smoothed_adj'] = ts_metrics(test_labels, point_adjustment(test_labels, s_scores))
    results['smoothed_scores'] = s_scores.tolist()
    results['radiis'] = radiis.tolist()

    # Save results to JSON file
    logger.info(f'Saving results to {exp_folder}/results.json')
    with open(f"{exp_folder}/results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Experiment completed: {args.exp_name}")

if __name__ == '__main__':
    main()
