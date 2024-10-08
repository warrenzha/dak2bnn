from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler

import numpy as np


def entropy(prob):
    return -1 * np.sum(prob * np.log(prob + 1e-15), axis=-1)


def predictive_entropy(mc_preds):
    """
    Compute the entropy of the mean of the predictive distribution
    obtained from Monte Carlo sampling during prediction phase.
    """
    return entropy(np.mean(mc_preds, axis=0))


def mutual_information(mc_preds):
    """
    Compute the difference between the entropy of the mean of the
    predictive distribution and the mean of the entropy.
    """
    mutual_info = entropy(np.mean(mc_preds, axis=0)) - np.mean(entropy(mc_preds),
                                                               axis=0)
    return mutual_info


def get_rho(sigma, delta):
    """
    sigma is represented by softplus function  'sigma = log(1 + exp(rho))' to make sure it
    remains always positive and non-transformed 'rho' gets updated during backprop.
    """
    rho = torch.log(torch.expm1(delta * torch.abs(sigma)) + 1e-20)
    return rho


def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if a < py_value[i] <= b:
                bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if bm[m] != 0:
            acc[m] = acc[m] / bm[m]
            conf[m] = conf[m] / bm[m]
    ece = 0
    for m in range(n_bins):
        ece += bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(bm)


def expected_calibration_error(y_true, y_prob, n_bins=15, return_bin_info=False):
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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class WarmUpLR(LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class MinMaxNormalize:
    """
    Min-max normalization to [0,1]
    """

    def __call__(self, sample):
        return (sample - sample.min()) / (sample.max() - sample.min())


class PrintOutput:
    """
    # A class to print the output to both terminal and file
    """

    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout

    def write(self, message):
        self.terminal.write(message)  # print to the terminal
        self.file.write(message)  # write to the file

    def flush(self):
        self.terminal.flush()
        self.file.flush()
