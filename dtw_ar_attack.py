import torch
import random
from tqdm import tqdm
import numpy as np

class DTWARAttacker:
    def __init__(self, model, seg_size, channel_nb, class_nb):
        self.net = model
        self.seg_size = seg_size
        self.channel_nb = channel_nb
        self.class_nb = class_nb

    def net_logits(self, x):
        x_output = self.net(x)
        rep = x_output[0]
        rep_dup = x_output[1]
        
        # Keep as tensors instead of converting to numpy
        dis = torch.sum((rep - self.c) ** 2, dim=1)
        dis2 = torch.sum((rep_dup - self.c) ** 2, dim=1)
        dis = dis + dis2
        
        # Create padding with tensors
        if x.shape[0] > dis.shape[0]:
            padding = torch.zeros(x.shape[0] - dis.shape[0], device=dis.device)
            dis_pad = torch.cat([padding, dis])
        else:
            dis_pad = dis
        
        # Convert threshold to tensor if it's not already
        if not isinstance(self.threshold, torch.Tensor):
            threshold = torch.tensor(self.threshold, dtype=dis_pad.dtype, device=dis_pad.device)
        else:
            threshold = self.threshold.to(device=dis_pad.device)
        
        # Calculate logits using tensors
        logits_1 = dis_pad - threshold
        logits_0 = -logits_1
        
        # Stack using torch.stack instead of np.vstack
        logits = torch.stack([logits_0, logits_1], dim=1)
        
        return logits


    def adv_loss_fn(self, X, t, rho):
        logits = self.net_logits(X)
        max_other = torch.max(logits.masked_fill(
            torch.nn.functional.one_hot(t, num_classes=self.class_nb).bool(), float('-inf')
        ), dim=1).values
        target_logits = logits.gather(1, t.unsqueeze(1)).squeeze(1)
        return torch.maximum(max_other - target_logits, torch.tensor(rho, dtype=logits.dtype, device=logits.device))

    def dtw_differentiable(self, path, x, y, torch_norm=2):
        """
        Calculate DTW distance between x and y using a predetermined path.
        
        Args:
            path: Tuple of (x_path, y_path) containing indices
            x: Source tensor
            y: Target tensor
            torch_norm: Norm for distance calculation
            
        Returns:
            DTW distance
        """
        if isinstance(path[0], torch.Tensor):
            x_path = path[0].to(device=x.device, dtype=torch.long)
            y_path = path[1].to(device=y.device, dtype=torch.long)
        else:
            x_path = torch.tensor(path[0], dtype=torch.long, device=x.device)
            y_path = torch.tensor(path[1], dtype=torch.long, device=y.device)
            
        if len(x_path) != len(y_path):
            raise ValueError("Error in DTW path length")
        
        # Handle single-example batch case (when x and y have shape [1, seq_len, features])
        distances = []
        if x.size(0) == 1:
            x_flat = x.squeeze(0)  # Remove batch dimension for indexing
            y_flat = y.squeeze(0)
            
            # Collect individual distances without in-place operations
            for i in range(len(x_path)):
                distances.append(torch.norm(x_flat[x_path[i]] - y_flat[y_path[i]], p=torch_norm))
        else:
            # Original behavior for multiple examples
            for i in range(len(x_path)):
                distances.append(torch.norm(x[x_path[i]] - y[y_path[i]], p=torch_norm))
        
        # Sum the distances without in-place operations
        return torch.sum(torch.stack(distances))

    def path_conversion(self, p, mat):
        dict_el = {(int(mat[i, j])): (i, j) for i in range(mat.shape[0]) for j in range(mat.shape[1])}
        dtw_path0 = [dict_el[int(el)][0] for el in p]
        dtw_path1 = [dict_el[int(el)][1] for el in p]
        return torch.tensor(dtw_path0, dtype=torch.long), torch.tensor(dtw_path1, dtype=torch.long)

    def forbidden_list_gen(self, xsi, mat):
        m, n = mat.shape
        forbidden_list = []
        seed = xsi
        for col in range(m):
            start = seed + col * (m + 1)
            end = (col + 1) * m
            forbidden_list.extend(range(start, end + 1))
        seed2 = m ** 2 - seed + 1
        for col in range(m, 0, -1):
            start = seed2 - (m - col) * (m + 1)
            end = m ** 2 - (m - col + 1) * m + 1
            forbidden_list.extend(range(start, end - 1, -1))
        # Add device parameter to ensure tensor is on the same device as mat
        return torch.tensor(forbidden_list, dtype=torch.long, device=mat.device)

    def dtw_random_path(self, xsi, mat, prob=[0.33, 0.33, 0.34]):
        m, n = mat.shape
        i, j = 0, 0
        path = [mat[i, j]]
        forbidden_list = self.forbidden_list_gen(xsi, mat)

        while i != m - 1 and j != n - 1:
            step = random.choices([1, 2, 3], weights=prob, k=1)[0]
            if step == 1 and i + 1 < m:
                if torch.isin(mat[i + 1, j], forbidden_list):
                    continue
                i += 1
            elif step == 2 and j + 1 < n:
                if torch.isin(mat[i, j + 1], forbidden_list):
                    continue
                j += 1
            elif step == 3 and i + 1 < m and j + 1 < n:
                i += 1
                j += 1
            path.append(mat[i, j])

        while i < m - 1:
            i += 1
            path.append(mat[i, j])
        while j < n - 1:
            j += 1
            path.append(mat[i, j])

        return path

    def dtwar_gradient_step(self, X_adv, X_orig, t, path, max_l2_loss, alpha, alpha_l2, rho):
        # Make sure t is the right type (long for one_hot encoding)
        if not isinstance(t, torch.Tensor) or t.dtype != torch.long:
            t = torch.tensor(t, dtype=torch.long, device=X_adv.device)
        
        # Ensure X_adv requires gradients
        X_adv = X_adv.detach().clone()
        X_adv.requires_grad_(True)
        
        # Ensure max_l2_loss is a tensor with the right device
        if not isinstance(max_l2_loss, torch.Tensor):
            max_l2_loss = torch.tensor(max_l2_loss, dtype=X_adv.dtype, device=X_adv.device)
            
        # Calculate DTW distances while preserving gradients
        if X_orig.shape[0] > 1:
            # Handle batch case
            dtw_dists = []
            for i in range(X_orig.shape[0]):
                dtw_dist = self.dtw_differentiable(path, X_orig[i:i+1], X_adv[i:i+1])
                dtw_dists.append(dtw_dist)
            dtw_dists = torch.stack(dtw_dists)
        else:
            # Single example case, avoid stack operation
            dtw_dists = self.dtw_differentiable(path, X_orig, X_adv).unsqueeze(0)
        
        # Calculate L2 distances
        l2_dists = torch.nn.functional.mse_loss(X_orig, X_adv, reduction='none').sum(dim=[1, 2]) / 2
        l2_dists = torch.minimum(l2_dists, max_l2_loss)
        
        # Get logits from model - this needs to maintain gradients
        logits = self.net_logits(X_adv)
        
        # Calculate target losses
        one_hot_mask = torch.nn.functional.one_hot(t, num_classes=self.class_nb).bool()
        masked_logits = logits.masked_fill(one_hot_mask, float('-inf'))
        max_other = torch.max(masked_logits, dim=1).values
        target_logits = logits.gather(1, t.unsqueeze(1)).squeeze(1)
        
        # Use full_like to avoid potential gradient issues
        rho_tensor = torch.full_like(target_logits, rho)
        target_losses = torch.maximum(max_other - target_logits, rho_tensor)
        
        # Calculate total loss
        dist_losses = alpha_l2 * l2_dists + alpha * dtw_dists
        
        # Ensure the loss is a scalar for backward
        total_loss = torch.mean(dist_losses + target_losses)
        
        # Compute gradients
        total_loss.backward()
        G = X_adv.grad.detach()
        
        return G, target_losses.detach(), dist_losses.detach(), dtw_dists.detach()

    def dtwar_attack(self, X, t, path=None, alpha=0.1, beta=0.1, eta=1e-2, rho=-5,
                     max_iter=1e3, delta_l2_loss=1, dtw_path_tightness=10, max_dtw_budget=None):
        batch_size = X.shape[0]

        min_X = 1.25 * torch.amin(X, dim=1)
        max_X = 1.25 * torch.amax(X, dim=1)
        max_l2_loss = torch.nn.functional.mse_loss(X, X + delta_l2_loss, reduction='none').sum(dim=[1, 2]) / 2

        noise = torch.randn_like(X) * eta
        X_adv = X + noise

        CT_dtw_mat = torch.arange(1, self.seg_size ** 2 + 1, device=X.device).view(self.seg_size, self.seg_size)
        if path is None:
            path = self.path_conversion(self.dtw_random_path(dtw_path_tightness, CT_dtw_mat), CT_dtw_mat)

        min_loss = torch.full((batch_size,), float('inf'), device=X.device)
        safe_Xadv = torch.zeros_like(X)

        for _ in range(int(max_iter)):
            G, target_losses, dist_losses, dtw_dists = self.dtwar_gradient_step(X_adv, X, t, path, max_l2_loss, alpha, -beta, rho)

            if max_dtw_budget is not None and (dtw_dists > max_dtw_budget).any():
                mask = dtw_dists <= max_dtw_budget
                if safe_Xadv is not None and mask.any():
                    return torch.where(mask.view(-1, 1, 1), safe_Xadv, X)
                else:
                    return X

            update_mask = (target_losses < 0) & (dist_losses < min_loss)
            safe_Xadv = torch.where(update_mask.view(-1, 1, 1), X_adv.detach(), safe_Xadv)
            min_loss = torch.where(update_mask, dist_losses, min_loss)

            if ((target_losses <= rho) & (dist_losses < min_loss)).all():
                return X_adv.detach()

            if torch.isnan(G).any():
                break

            X_adv = X_adv - eta * G
            X_adv = torch.max(torch.min(X_adv, max_X.unsqueeze(1)), min_X.unsqueeze(1))

        final_logits = self.net_logits(X_adv)
        final_target_losses = torch.maximum(
            torch.max(final_logits.masked_fill(
                torch.nn.functional.one_hot(t, num_classes=self.class_nb).bool(), float('-inf')
            ), dim=1).values - final_logits.gather(1, t.unsqueeze(1)).squeeze(1),
            rho * torch.ones_like(t, dtype=final_logits.dtype, device=final_logits.device)
        )

        if (final_target_losses >= 0).any():
            return torch.where(final_target_losses.view(-1, 1, 1) < 0, X_adv, safe_Xadv)
        else:
            return X_adv
        
    def gen_adv_test_sub_seqs(self, X_test, test_labels, batch_size, c, threshold, window_size, dtw_budget, device):
    
        from deepod.models.time_series.couta import _SubseqData
        from torch.utils.data import DataLoader
        from deepod.utils.utility import get_sub_seqs, get_sub_seqs_label
        
        self.c = c
        self.threshold = threshold

        test_sub_seqs = get_sub_seqs(X_test, seq_len=self.seg_size, stride=1)
        test_labels_sub_seqs = get_sub_seqs_label(test_labels, seq_len=self.seg_size, stride=1)
        test_dataset = _SubseqData(test_sub_seqs, test_labels_sub_seqs)
        dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
        adv_test_sub_seqs = []

        self.net.eval()

        for x, y in tqdm(dataloader):
            x = x.float().to(device)
            y = y.long().to(device)
            x_adv = self.dtwar_attack(x, y, path=None, alpha=0.1, beta=0.1, eta=1e-10, rho=-5,
                    max_iter=1e1, delta_l2_loss=1, dtw_path_tightness=window_size, max_dtw_budget=dtw_budget)
            adv_test_sub_seqs.append(x_adv.cpu().numpy())

        adv_test_sub_seqs = np.concatenate(adv_test_sub_seqs, axis=0)
        assert adv_test_sub_seqs.shape[0] == test_sub_seqs.shape[0], "The number of adversarial examples does not match the number of subsequences."
        return adv_test_sub_seqs

