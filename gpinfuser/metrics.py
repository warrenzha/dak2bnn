import math
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from torch import Tensor
from torch.distributions import Categorical
from gpytorch.distributions import MultivariateNormal

def mean_squared_error(y: Tensor, y_hat: Tensor) -> float:
    return (y - y_hat).square().mean().item()

def root_mean_squared_error(y: Tensor, y_hat: Tensor) -> float:
    return math.sqrt(mean_squared_error(y, y_hat))

def reg_negative_log_likelihood(y: Tensor, means: Tensor, variances: Tensor, *args, **kwargs) -> float:
    nll = 0.5 * (math.log(2 * math.pi) + variances.log() + (means - y).square()/variances)
    return nll.mean().item()

def accuracy_score(y: Tensor, y_pred: Tensor, *args, **kwargs) -> float:
    return y.eq(y_pred).float().mean().item()

def topk_accuracy_score(y: Tensor, y_prob: Tensor, k: int=5, *args, **kwargs) -> float:
    if y.ndim == 2:
        y = y.argmax(1)
    pred = y_prob.topk(k, dim=1, largest=True, sorted=True).indices.t()
    correct = pred.eq(y[None])
    return correct.flatten().sum().div(len(y)).item()

def brier_score(y: Tensor, y_prob: Tensor, *args, **kwargs):
    y, y_prob = y.cpu(), y_prob.cpu()
    if (y.ndim == 1) and (y_prob.ndim > 1):
        y = F.one_hot(y.long(), y_prob.size(1))
        return (y - y_prob).square().sum(1).mean().item()
    return (y - y_prob).square().mean().item()

def clf_negative_log_likelihood(y: Tensor, y_prob: Tensor, *args, **kwargs) -> float:
    if y_prob.ndim == 1:
        return F.binary_cross_entropy(y_prob, y, reduction='mean').item()
    if y.ndim == 1:
        y = F.one_hot(y.long(), y_prob.size(1))
    return -(y * y_prob.log()).sum(-1).mean().item()

def expected_calibration_error(y_true: Tensor, y_prob: Tensor, n_bins=15, return_bin_info=False):
    y_true, y_prob = y_true.cpu(), y_prob.cpu()
    if y_prob.ndim == 1:
        confs, accs = y_prob, y_true
    elif y_prob.ndim == 2:
        confs, y_pred = y_prob.max(dim=1)
        accs = y_true.eq(y_pred).float()
    else:
        raise ValueError(f'y_prob shape `{y_prob.shape}` is not valid')
    
    bin_bounds = torch.linspace(0, 1, n_bins + 1)
    bin_indices = torch.bucketize(confs, bin_bounds[1:-1])
    bin_counts = torch.bincount(bin_indices, minlength=n_bins)
    bin_confs = torch.bincount(bin_indices, weights=confs, minlength=n_bins) / bin_counts
    bin_accs = torch.bincount(bin_indices, weights=accs, minlength=n_bins) / bin_counts
    bin_confs[bin_confs.isnan()] = 0
    bin_accs[bin_accs.isnan()] = 0
    ece = ((bin_accs - bin_confs).abs().double() @ bin_counts.double()).item() / len(y_true)
    
    if return_bin_info:
        return ece, bin_accs, bin_confs, bin_counts
    
    return ece

def reliability_diagram(bin_accs, bin_confs, figsize=[8, 4]):
    """
    Shows the reliability diagram of the calibration of a set of predicted probabilities.
    
    Parameters:
        bin_accs (list) : A list of bin accuracies
        bin_confs (list) : A list of bin confidences
        
    Returns:
        ax: The plot object
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot([0, 1], [0, 1], '--k', label='Perfect Calibration')
    
    bin_edges = np.linspace(0, 1, len(bin_accs) + 1)
    bin_diffs = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + bin_diffs/2
    ax.bar(
        bin_centers, bin_confs, width=bin_diffs, alpha=0.5,
        linewidth=1, edgecolor='k', label='Confidence'
    )
    ax.bar(
        bin_centers, bin_accs, color='r', width=bin_diffs, alpha=0.5,
        linewidth=1, edgecolor='k', label='Accuracy'
    )
    
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('Reliability Diagram')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend()
    fig.tight_layout()

    return fig, ax

# GP-based models
class GPRMetrics:
    def __str__(self):
        return 'metrics: mse, rmse, nll'
    
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def rmse(y: Tensor, f: MultivariateNormal) -> float:
        y = y.squeeze().cpu()
        y_pred = f.mean.squeeze().cpu()
        return root_mean_squared_error(y, y_pred)

    @staticmethod
    def nll(y: Tensor, f: MultivariateNormal) -> float:
        y = y.squeeze().cpu()
        mean = f.mean.squeeze().cpu()
        variance = f.variance.squeeze().cpu()
        return reg_negative_log_likelihood(y, mean, variance)

class GPCMetrics:
    def __str__(self):
        return 'metrics: accuracy, nll'
    
    def __repr__(self):
        return self.__str__()
    
    @staticmethod
    def accuracy(y: Tensor, f: Any) -> float:
        y = y.squeeze().cpu()
        y_prob = f.probs.squeeze().cpu()
        if y_prob.ndim >= 2:
            y_prob = y_prob.mean(0)
        y_pred = y_prob > 0.5 if y_prob.ndim == 1 else y_prob.argmax(-1)
        return accuracy_score(y, y_pred)

    @staticmethod
    def nll(y: Tensor, f: Any) -> float:
        y = y.squeeze().cpu()
        y_prob = f.probs.squeeze().cpu()
        if y_prob.ndim >= 2:
            y_prob = y_prob.mean(0)
        return clf_negative_log_likelihood(y, y_prob)
    
    @staticmethod
    def ece(y: Tensor, f: Any) -> float:
        y = y.squeeze().cpu()
        y_prob = f.probs.squeeze().cpu()
        if y_prob.ndim >= 2:
            y_prob = y_prob.mean(0)
        return expected_calibration_error(y, y_prob)

gpr_metrics = GPRMetrics()
gpc_metrics = GPCMetrics()