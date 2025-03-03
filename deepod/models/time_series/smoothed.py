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


class SmoothedMedian(nn.Module):
    def __init__(self, base_model, sigma):
        super(SmoothedMedian, self).__init__()
        self.base_model = base_model
        self.sigma = sigma

    def decision_function(self, X, return_rep=False):

        testing_n_samples = X.shape[0]
        seqs = get_sub_seqs(X, seq_len=self.base_model.seq_len, stride=1)
        dataloader = DataLoader(seqs, batch_size=self.base_model.batch_size,
                                shuffle=False, drop_last=False)
        
        scores_list = []
        for batch_x in tqdm(dataloader):  # test_set
            scores = self.base_model.decision_function(batch_x, get_subseqs=False)
            scores_list.append(scores)
        scores = np.concatenate(scores_list)

        padding = np.zeros(testing_n_samples - scores.shape[0])
        scores = np.hstack((padding, scores))
        
        return scores


    def predict_range(self, x: torch.tensor, n: int, batch_size: int, q_u: int, q_l: int):

        input_imgs = x.repeat((batch_size, 1, 1, 1))
        for i in range(n//batch_size):
            # Get detections
            with torch.no_grad():
                detections = self.base_model(input_imgs + torch.randn_like(input_imgs) * self.sigma)
                # detections, _ = non_max_suppression(detections, conf_thres, nms_thres)
                self.detection_acc.track(detections)

        self.detection_acc.tensorize()
        detections = [self.detection_acc.median()]
        detections_l = [self.detection_acc.k(q_l)]
        detections_u = [self.detection_acc.k(q_u)]
        self.detection_acc.clear()
        return detections, detections_u, detections_l