import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
import math
import time
from deepod.utils.utility import get_sub_seqs
from deepod.core.base_model import BaseDeepAD
from tqdm import tqdm
from scipy.stats import norm

ZERO_PROB_REPLACEMENT = 1e-6
ONE_PROB_REPLACEMENT = 1 - ZERO_PROB_REPLACEMENT


class SmoothedMedian(nn.Module):
    def __init__(self, base_model):
        super(SmoothedMedian, self).__init__()
        self.base_model = base_model
        self.training_anomaly_score_threshold = self.base_model.threshold_

        self.base_model.verbose = 1
        

    def decision_function(self, X, sigma, smooth_count, window_size, threshold=None, saved_noise_scores=None):
        if threshold is None:
            threshold = self.training_anomaly_score_threshold
            print(f"Using training-time Anomaly Score Threshold: {threshold}")
        else:
            print(f"Using custom Anomaly Score Threshold: {threshold}")
        if saved_noise_scores is None:
            saved_noise_scores =[]
            existing_noise_scores = False
        else:
            existing_noise_scores = True
            print(f"Using existing batch noise scores list of length {len(saved_noise_scores)}")

        testing_n_samples = X.shape[0]
        seqs = get_sub_seqs(X, seq_len=self.base_model.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.base_model.batch_size,
                                shuffle=False, drop_last=False)
        scores_list = []
        radii_list = []
        for i, batch_x in enumerate(tqdm(dataloader)):
            if existing_noise_scores:
                batch_noise_scores = saved_noise_scores[i]
            # get the smoothed median score here and the radius
            else:
                batch_noise_scores = []
                for i in range(smooth_count):
                    with torch.no_grad():
                        noise_batch_x = batch_x + torch.randn_like(batch_x) * sigma
                        noise_scores = self.base_model.decision_function(noise_batch_x, get_subseqs=False)
                        batch_noise_scores.append(noise_scores)
                batch_noise_scores = np.stack(batch_noise_scores, axis=0) # shape [smooth_count, batch_size]
                saved_noise_scores.append(batch_noise_scores)

            scores, radii = self.dtw_certify(batch_x, batch_noise_scores, sigma, window_size, threshold)
        
            scores_list.append(scores)
            radii_list.append(radii)    

        scores = np.concatenate(scores_list)
        radii = np.concatenate(radii_list)

        padding = np.zeros(testing_n_samples - scores.shape[0])
        scores = np.hstack((padding, scores))
        radii = np.hstack((padding, radii))
        
        return scores, radii, saved_noise_scores

    
    def dtw_slack(self, batch_x, w):
        batch_size, seq_len, n_channels = batch_x.shape
        device = batch_x.device
        
        # Create indices matrix for all possible pairs
        row_indices = torch.arange(seq_len, device=device).view(1, -1, 1).expand(batch_size, seq_len, seq_len)
        col_indices = torch.arange(seq_len, device=device).view(1, 1, -1).expand(batch_size, seq_len, seq_len)
        
        # Create mask for values within window
        mask = (col_indices >= (row_indices - w)) & (col_indices <= (row_indices + w))
        
        # Expand batch_x for broadcasting: [batch_size, seq_len, 1, n_channels]
        points1 = batch_x.unsqueeze(2)
        
        # Expand batch_x for broadcasting: [batch_size, 1, seq_len, n_channels]
        points2 = batch_x.unsqueeze(1)
        
        # Compute all pairwise distances: [batch_size, seq_len, seq_len]
        all_distances = torch.sqrt(torch.sum((points1 - points2) ** 2, dim=3)) / n_channels
        # all_distances = torch.sum(torch.abs((points1 - points2)), dim=3) / n_channels
        
        # Mask out distances outside window and get max for each point
        masked_distances = all_distances.masked_fill(~mask, -float('inf'))
        max_distances = masked_distances.max(dim=2)[0]
        
        # Compute M and R2
        M = torch.max(max_distances, dim=1)[0]
        R2 = torch.sum(max_distances ** 2, dim=1)
        
        return M, R2
    

    def dtw_slack_old(self, batch_x, w):
        batch_size, seq_len, n_channels = batch_x.shape
        
        # Pre-allocate tensor for max distances
        max_distances = torch.zeros(batch_size, seq_len, device=batch_x.device)
        
        # Convert to contiguous tensor for better performance
        batch_x = batch_x.contiguous()
        
        # Compute all pairwise distances efficiently using matrix operations
        for t in range(seq_len):
            # Calculate window bounds
            left_bound = max(0, t - w)
            right_bound = min(seq_len, t + w + 1)
            
            # Extract current point for all batches: [batch_size, 1, n_channels]
            current_points = batch_x[:, t:t+1, :]
            
            # Extract window points for all batches: [batch_size, window_size, n_channels]
            window_points = batch_x[:, left_bound:right_bound, :]
            
            # Compute L2 distances between current point and all window points
            # Broadcasting happens automatically: [batch_size, window_size]
            distances = torch.sqrt(torch.sum((current_points - window_points) ** 2, dim=2)) / n_channels
            
            # Get max distance for each batch at this time step
            max_distances[:, t] = torch.max(distances, dim=1)[0]
        
        # Compute M and R2 as required
        M = torch.max(max_distances, dim=1)[0]  # Maximum distance for each batch
        R2 = torch.sum(max_distances ** 2, dim=1)  # Sum of squared distances for each batch
        
        return M, R2

    def dtw_certify(self, batch_x, batch_noise_scores, sigma, window_size, anomaly_score_threshold):
        batch_noise_scores = batch_noise_scores.T  # shape [batch_size, smooth_count]
        smooth_count = batch_noise_scores.shape[1]
        
        # Calculate median for each row
        medians = np.median(batch_noise_scores, axis=1)
        
        # Sort each row
        sorted_scores = np.sort(batch_noise_scores, axis=1)
        
        # Initialize percentile array
        percentiles = np.zeros_like(medians)
        
        # For rows where median > threshold
        above_threshold_mask = medians > anomaly_score_threshold
        if np.any(above_threshold_mask):
            # For each row where median > threshold, find lowest index where score > threshold
            for i in np.where(above_threshold_mask)[0]:
                # Find first index where sorted score > threshold
                indices = np.where(sorted_scores[i] > anomaly_score_threshold)[0]
                if len(indices) > 0:
                    lowest_idx = indices[0]
                    # Convert to percentile (0-100)
                    percentiles[i] = lowest_idx / smooth_count
                else:
                    percentiles[i] = 0.5  # All scores are below threshold
        
        # For rows where median <= threshold
        below_threshold_mask = ~above_threshold_mask
        if np.any(below_threshold_mask):
            # For each row where median <= threshold, find highest index where score < threshold
            for i in np.where(below_threshold_mask)[0]:
                # Find last index where sorted score < threshold
                indices = np.where(sorted_scores[i] < anomaly_score_threshold)[0]
                if len(indices) > 0:
                    highest_idx = indices[-1]
                    # Convert to percentile (0-100)
                    percentiles[i] = (highest_idx + 1) / smooth_count
                else:
                    percentiles[i] = 0.5  # All scores are above threshold
        
        # Replace extreme values to avoid numerical issues with norm.ppf
        percentiles[percentiles == 0] = ZERO_PROB_REPLACEMENT
        percentiles[percentiles == 1] = ONE_PROB_REPLACEMENT
    
        r = sigma * np.abs(norm.ppf(percentiles))

        M, R2 = self.dtw_slack(batch_x, w=window_size)
        
        # Convert torch tensors to numpy arrays if needed
        if isinstance(M, torch.Tensor):
            M = M.cpu().numpy()
        if isinstance(R2, torch.Tensor):
            R2 = R2.cpu().numpy()
        
        # Check if r^2 <= R2, set radius to 0.0 in that case
        # Otherwise calculate the radius using the formula
        condition = r**2 <= R2
        radii = np.zeros_like(M)
        valid_indices = ~condition
        if np.any(valid_indices):
            radii[valid_indices] = np.sqrt(M[valid_indices]**2 + r[valid_indices]**2 - R2[valid_indices]) - M[valid_indices]
        radii[condition] = 0.0

        return medians, radii
