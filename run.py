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
from deepod.metrics import ts_metrics, point_adjustment, get_best_f1_and_threshold
from analyse import certified_f1_p_r, radii_stats, create_range, certified_stats

def none_or_str(value):
    if value.lower() == 'none':
        return None
    return value

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
    parser.add_argument('--load_model', type=none_or_str, default=None, help='Path to saved model to load (default: None)')
    parser.add_argument('--load_noise_scores', type=none_or_str, default=None, help='Path to saved noise scores to load (default: None)')
    parser.add_argument('--save_model', type=bool, default=False, help='Save model after training (default: False)')

    # Common model parameters
    parser.add_argument('--seq_len', type=int, default=10, help='Sequence length')
    parser.add_argument('--stride', type=int, default=1, help='Stride')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--epoch_steps', type=int, default=-1, help='Number of steps per epoch (-1 means all data)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    # Smooth parameters
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma for smoothing')
    parser.add_argument('--window_size', type=int, default=2, help='Window size for smoothing')
    parser.add_argument('--smooth_count', type=int, default=2000, help='Number of times to smooth')
    
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
    
    test_labels = np.load(data_test_label_path, allow_pickle=True)
    X_train_df = pd.DataFrame(np.load(data_train_path, allow_pickle=True))
    X_test_df = pd.DataFrame(np.load(data_test_path, allow_pickle=True))
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
    results['clean_metrics'] = ts_metrics(test_labels, scores)
    results['clean_adj_metrics'] = ts_metrics(test_labels, point_adjustment(test_labels, scores))

    logger.info('Collecting smoothed results...')
    smoothed_clf = SmoothedMedian(clf)

    if args.load_noise_scores is not None:
        logger.info(f'Loading noise scores from {args.load_noise_scores}')
        with open(args.load_noise_scores, 'rb') as f:
            saved_noise_scores = pickle.load(f)
    else:
        saved_noise_scores = None

    scores, _, saved_noise_scores = smoothed_clf.decision_function(X_test, args.sigma, args.smooth_count, args.window_size, saved_noise_scores=saved_noise_scores)
    
    if args.load_noise_scores is None:
        # Save batch noise scores
        logger.info(f'Saving noise scores to {exp_folder}/saved_noise_scores.pkl')
        with open(f"{exp_folder}/saved_noise_scores.pkl", 'wb') as f:
            pickle.dump(saved_noise_scores, f)
    
    radii_thresholds = create_range(0.0, 0.5, 0.005)
    results['radii_thresholds'] = radii_thresholds

    results['smoothed_metrics'] = ts_metrics(test_labels, scores)
    _, _, _, score_threshold = get_best_f1_and_threshold(test_labels, scores)
    _, radiis, _ = smoothed_clf.decision_function(X_test, args.sigma, args.smooth_count, args.window_size, threshold=score_threshold, saved_noise_scores=saved_noise_scores)
    
    f1, p, r = certified_f1_p_r(test_labels, scores, radiis, radii_thresholds, score_threshold)
    results['certified_stats'] = certified_stats(test_labels, scores, radiis, radii_thresholds, score_threshold)
    
    results['smoothed_adj_metrics'] = ts_metrics(test_labels, point_adjustment(test_labels, scores))
    _, _, _, score_threshold = get_best_f1_and_threshold(test_labels, point_adjustment(test_labels, scores))
    _, radiis, _ = smoothed_clf.decision_function(X_test, args.sigma, args.smooth_count, args.window_size, threshold=score_threshold, saved_noise_scores=saved_noise_scores)

    f1, p, r = certified_f1_p_r(test_labels, scores, radiis, radii_thresholds, score_threshold, point_adj=True)
    results['certified_adj_stats'] = certified_stats(test_labels, scores, radiis, radii_thresholds, score_threshold, point_adj=True)

    # Save results to JSON file
    logger.info(f'Saving results to {exp_folder}/results.json')
    with open(f"{exp_folder}/results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Experiment completed: {args.exp_name}")

if __name__ == '__main__':
    main()
