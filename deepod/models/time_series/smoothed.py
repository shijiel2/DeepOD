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

ZERO_PROB_REPLACEMENT = 1e-5
ONE_PROB_REPLACEMENT = 1 - ZERO_PROB_REPLACEMENT


class SmoothedMedian(nn.Module):
    def __init__(self, base_model, sigma, smooth_count=100):
        super(SmoothedMedian, self).__init__()
        self.base_model = base_model
        self.sigma = sigma
        self.smooth_count = smooth_count
        self.anomaly_score_threshold = self.base_model.threshold_

        self.base_model.verbose = 1
        print(f"Anomaly score threshold: {self.anomaly_score_threshold}")

    def decision_function(self, X, return_rep=False):

        testing_n_samples = X.shape[0]
        seqs = get_sub_seqs(X, seq_len=self.base_model.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.base_model.batch_size,
                                shuffle=False, drop_last=False)
        
        scores_list = []
        radii_list = []
        for batch_x in tqdm(dataloader):
            # get the smoothed median score here and the radius
            batch_noise_scores = []
            for i in range(self.smooth_count):
                with torch.no_grad():
                    noise_batch_x = batch_x + torch.randn_like(batch_x) * self.sigma
                    noise_scores = self.base_model.decision_function(noise_batch_x, get_subseqs=False)
                    batch_noise_scores.append(noise_scores)
            
            batch_noise_scores = np.stack(batch_noise_scores, axis=0) # shape [smooth_count, batch_size]
            scores, radii = self.certify(batch_noise_scores, batch_x)
            
            scores_list.append(scores)
            radii_list.append(radii)

        scores = np.concatenate(scores_list)
        radii = np.concatenate(radii_list)

        padding = np.zeros(testing_n_samples - scores.shape[0])
        scores = np.hstack((padding, scores))
        radii = np.hstack((padding, radii))
        
        return scores, radii

    
    def dtw_slack(self, batch_x, w):
        batch_size, seq_len, n_channels = batch_x.shape
        
        # Initialize tensors to store distances
        max_distances = torch.zeros(batch_size, seq_len)
        
        # For each time step
        for t in range(seq_len):
            # Calculate window bounds
            left_bound = max(0, t - w)
            right_bound = min(seq_len, t + w + 1)
            
            # For each batch
            for b in range(batch_size):
                current_point = batch_x[b, t, :]  # Current point at time t
                max_dist = 0.0
                
                # Compare with all points in the window
                for j in range(left_bound, right_bound):
                    window_point = batch_x[b, j, :]
                    # Compute L2 distance between current point and window point
                    dist = torch.norm(current_point - window_point, p=2) / n_channels
                    max_dist = max(max_dist, dist)
                
                # Store the max distance for this batch and time step
                max_distances[b, t] = max_dist
        
        # Compute M and R2 as required
        M = torch.max(max_distances, dim=1)[0]  # Maximum distance for each batch
        R2 = torch.sum(max_distances ** 2, dim=1)  # Sum of squared distances for each batch
        
        return M, R2
    

    def certify(self, batch_noise_scores, batch_x):
        batch_noise_scores = batch_noise_scores.T  # shape [batch_size, smooth_count]
        
        # Calculate median for each row
        medians = np.median(batch_noise_scores, axis=1)
        
        # Sort each row
        sorted_scores = np.sort(batch_noise_scores, axis=1)
        
        # Initialize percentile array
        percentiles = np.zeros_like(medians)
        
        # For rows where median > threshold
        above_threshold_mask = medians > self.anomaly_score_threshold
        if np.any(above_threshold_mask):
            # For each row where median > threshold, find lowest index where score > threshold
            for i in np.where(above_threshold_mask)[0]:
                # Find first index where sorted score > threshold
                indices = np.where(sorted_scores[i] > self.anomaly_score_threshold)[0]
                if len(indices) > 0:
                    lowest_idx = indices[0]
                    # Convert to percentile (0-100)
                    percentiles[i] = lowest_idx / self.smooth_count
                else:
                    percentiles[i] = 0.5  # All scores are below threshold
        
        # For rows where median <= threshold
        below_threshold_mask = ~above_threshold_mask
        if np.any(below_threshold_mask):
            # For each row where median <= threshold, find highest index where score < threshold
            for i in np.where(below_threshold_mask)[0]:
                # Find last index where sorted score < threshold
                indices = np.where(sorted_scores[i] < self.anomaly_score_threshold)[0]
                if len(indices) > 0:
                    highest_idx = indices[-1]
                    # Convert to percentile (0-100)
                    percentiles[i] = (highest_idx + 1) / self.smooth_count
                else:
                    percentiles[i] = 0.5  # All scores are above threshold
        
        # Replace extreme values to avoid numerical issues with norm.ppf
        percentiles[percentiles == 0] = ZERO_PROB_REPLACEMENT
        percentiles[percentiles == 1] = ONE_PROB_REPLACEMENT
    
        r = self.sigma * np.abs(norm.ppf(percentiles))

        M, R2 = self.dtw_slack(batch_x, w=2)

        radii = np.sqrt(M**2 + r**2 - R2) - M

        return medians, radii
