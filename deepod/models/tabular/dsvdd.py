# -*- coding: utf-8 -*-
"""
One-class classification
@Author: Hongzuo Xu <hongzuoxu@126.com, xuhongzuo13@nudt.edu.cn>
"""

from deepod.core.base_model import BaseDeepAD
from deepod.core.networks.base_networks import MLPnet
from deepod.metrics import tabular_metrics
from torch.utils.data import DataLoader
import torch
import time
from ray import tune
from ray.air import session, Checkpoint
from ray.tune.schedulers import ASHAScheduler
from functools import partial


class DeepSVDD(BaseDeepAD):
    """ Deep One-class Classification (Deep SVDD) for anomaly detection
    See :cite:`ruff2018deepsvdd` for details

    Parameters
    ----------
    epochs: int, optional (default=100)
        Number of training epochs

    batch_size: int, optional (default=64)
        Number of samples in a mini-batch

    lr: float, optional (default=1e-3)
        Learning rate

    rep_dim: int, optional (default=128)
        Dimensionality of the representation space

    hidden_dims: list, str or int, optional (default='100,50')
        Number of neural units in hidden layers
            - If list, each item is a layer
            - If str, neural units of hidden layers are split by comma
            - If int, number of neural units of single hidden layer

    act: str, optional (default='ReLU')
        activation layer name
        choice = ['ReLU', 'LeakyReLU', 'Sigmoid', 'Tanh']

    bias: bool, optional (default=False)
        Additive bias in linear layer

    epoch_steps: int, optional (default=-1)
        Maximum steps in an epoch
            - If -1, all the batches will be processed

    prt_steps: int, optional (default=10)
        Number of epoch intervals per printing

    device: str, optional (default='cuda')
        torch device,

    verbose: int, optional (default=1)
        Verbosity mode

    random_state： int, optional (default=42)
        the seed used by the random

    """
    def __init__(self, epochs=100, batch_size=64, lr=1e-3,
                 rep_dim=128, hidden_dims='100,50', act='ReLU', bias=False,
                 epoch_steps=-1, prt_steps=10, device='cuda',
                 verbose=2, random_state=42):
        super(DeepSVDD, self).__init__(
            model_name='DeepSVDD', data_type='tabular', epochs=epochs, batch_size=batch_size, lr=lr,
            network='MLP',
            epoch_steps=epoch_steps, prt_steps=prt_steps, device=device,
            verbose=verbose, random_state=random_state
        )

        self.hidden_dims = hidden_dims
        self.rep_dim = rep_dim
        self.act = act
        self.bias = bias

        self.c = None
        return

    def training_prepare(self, X, y):
        train_loader = DataLoader(X, batch_size=self.batch_size, shuffle=True)

        network_params = {
            'n_features': self.n_features,
            'n_hidden': self.hidden_dims,
            'n_output': self.rep_dim,
            'activation': self.act,
            'bias': self.bias
        }
        net = MLPnet(**network_params).to(self.device)

        self.c = self._set_c(net, train_loader)
        criterion = DSVDDLoss(c=self.c)

        if self.verbose >= 2:
            print(net)

        return train_loader, net, criterion

    def inference_prepare(self, X):
        test_loader = DataLoader(X, batch_size=self.batch_size,
                                 drop_last=False, shuffle=False)
        assert self.c is not None
        self.criterion = DSVDDLoss(c=self.c, reduction='none')
        return test_loader

    def training_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        z = net(batch_x)
        loss = criterion(z)
        return loss

    def inference_forward(self, batch_x, net, criterion):
        batch_x = batch_x.float().to(self.device)
        batch_z = net(batch_x)
        s = criterion(batch_z)
        return batch_z, s

    def _training_ray(self, config, X_test, y_test):
        train_data = self.train_data[:int(0.8 * len(self.train_data))]
        val_data = self.train_data[int(0.8 * len(self.train_data)):]

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        self.net = self.set_tuned_net(config)

        self.c = self._set_c(self.net, train_loader)
        criterion = DSVDDLoss(c=self.c, reduction='mean')

        optimizer = torch.optim.Adam(self.net.parameters(), lr=config['lr'], eps=1e-6)

        self.net.train()
        for i in range(config['epochs']):
            t1 = time.time()
            total_loss = 0
            cnt = 0
            for batch_x in train_loader:
                loss = self.training_forward(batch_x, self.net, criterion)
                self.net.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                cnt += 1

                # terminate this epoch when reaching assigned maximum steps per epoch
                if cnt > self.epoch_steps != -1:
                    break

            # validation phase
            val_loss = []
            with torch.no_grad():
                for batch_x in val_loader:
                    loss = self.training_forward(batch_x, self.net, criterion)
                    val_loss.append(loss)
            val_loss = torch.mean(torch.stack(val_loss)).data.cpu().item()

            test_metric = -1
            if X_test is not None and y_test is not None:
                scores = self.decision_function(X_test)
                test_metric = tabular_metrics(y_test, scores)[0]  # use adjusted Best-F1

            t = time.time() - t1
            if self.verbose >= 1 and (i == 0 or (i+1) % self.prt_steps == 0):
                print(f'epoch{i+1:3d}, '
                      f'training loss: {total_loss/cnt:.6f}, '
                      f'validation loss: {val_loss:.6f}, '
                      f'test F1: {test_metric:.3f},  '
                      f'time: {t:.1f}s')

            checkpoint_data = {
                "epoch": i,
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                'c': self.c
            }
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            session.report(
                {"loss": val_loss, "metric": test_metric},
                checkpoint=checkpoint,
            )

    def load_ray_checkpoint(self, best_config, best_checkpoint):
        self.net = self.set_tuned_net(best_config)
        self.net.load_state_dict(best_checkpoint['net_state_dict'])
        self.c = best_checkpoint['c']
        return

    def set_tuned_net(self, config):
        network_params = {
            'n_features': self.n_features,
            'n_hidden': config['hidden_dims'],
            'n_output': config['rep_dim'],
            'activation': self.act,
            'bias': self.bias
        }
        net = MLPnet(**network_params).to(self.device)
        return net

    @staticmethod
    def set_tuned_params():
        config = {
            'lr': tune.grid_search([1e-5, 1e-4, 1e-3, 1e-2]),
            'epochs': tune.grid_search([20, 50, 100]),
            'rep_dim': tune.grid_search([16, 64, 128, 512]),
            'hidden_dims': tune.choice(['100,100', '100'])
        }
        return config

    def _set_c(self, net, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        net.eval()
        z_ = []
        with torch.no_grad():
            for x in dataloader:
                x = x.float().to(self.device)
                z = net(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c


class DSVDDLoss(torch.nn.Module):
    """

    Parameters
    ----------
    c: torch.Tensor
        Center of the pre-defined hyper-sphere in the representation space

    reduction: str, optional (default='mean')
        choice = [``'none'`` | ``'mean'`` | ``'sum'``]
            - If ``'none'``: no reduction will be applied;
            - If ``'mean'``: the sum of the output will be divided by the number of
            elements in the output;
            - If ``'sum'``: the output will be summed

    """
    def __init__(self, c, reduction='mean'):
        super(DSVDDLoss, self).__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, rep, reduction=None):
        loss = torch.sum((rep - self.c) ** 2, dim=1)

        if reduction is None:
            reduction = self.reduction

        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        elif reduction == 'none':
            return loss
