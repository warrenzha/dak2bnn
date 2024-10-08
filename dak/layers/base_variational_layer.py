# BNN layers partially borrowed from BayesianTorch: https://github.com/IntelLabs/bayesian-torch
# ===============================================================================================


import torch
import torch.nn as nn
from itertools import repeat
import collections


def get_kernel_size(x, n):
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class _BaseVariationalLayer(nn.Module):
    """
    The base variational layer is implemented as a :class:`torch.nn.Module` that, when called on two distributions 
    :math:`Q` and :math:`P` returns a :obj:`torch.Tensor` that represents the KL divergence between two gaussians.
    """

    def __init__(self):
        super().__init__()
        self._dnn_to_bnn_flag = False

    @property
    def dnn_to_bnn_flag(self):
        return self._dnn_to_bnn_flag

    @dnn_to_bnn_flag.setter
    def dnn_to_bnn_flag(self, value):
        self._dnn_to_bnn_flag = value

    def kl_div(self, mu_q, sigma_q, mu_p, sigma_p):
        """
        Calculates kl divergence between two gaussians (Q || P).

        :param mu_q: mean of distribution Q
        :type mu_q: torch.Tensor
        :sigma_q: deviation of distribution Q
        :type sigma_q: torch.Tensor
        :mu_p: mean of distribution P
        :type mu_p: torch.Tensor
        :sigma_p: deviation of distribution P
        :type sigma_p: torch.Tensor

        :return: the KL divergence between Q and P.
        """
        kl = torch.log(sigma_p) - torch.log(
            sigma_q) + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * (sigma_p ** 2)) - 0.5
        return kl.mean()
