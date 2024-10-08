from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn


class LaplaceKernel(nn.Module):
    def __init__(self, lengthscale=1.0):
        super().__init__()
        self.theta = lengthscale

        self.lengthscale = nn.Parameter(torch.Tensor(1,))
        nn.init.constant_(self.lengthscale, 1.0)

    def forward(self, x1, x2=None, diag=False):
        # Size checking
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)  # Add a last dimension, if necessary
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        d = x1.shape[-1]
        lengthscale = x1.new_full(size=(d,), fill_value=self.theta, dtype=x1.dtype) * self.lengthscale

        adjustment = x1.mean(dim=-2, keepdim=True)  # [d] size tensor
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.sum(torch.abs(x1_-x2_), dim=-1)
        else:
            distance = torch.cdist(x1_, x2_, p=1)
            distance = distance.clamp_min(1e-15)

        res = torch.exp(-distance)
        return res


class LaplaceProductKernel(nn.Module):
    """
    Computes the covariance based on the Laplace product kernel.

    :param lengthscale: Set this if you want a customized lengthscale. (Default: 1.0.)
    :type lengthscale: float, optional
    """

    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, 
                diag: bool = False, **params) -> Tensor:
        """
        :param x1: First set of data of shape :math:`(n,d)`.
        :type x1: torch.Tensor.float
        :param x2: Second set of data of shape :math:`(m,d)`.
        :type x2: torch.Tensor.float
        :param diag: Compute diagonal covariance matrix if `True`. It must be the case that `x1 == x2`.
        :type diag: bool, optional
        
        :return: The kernel matrix or vector. The shape depends on the kernel's mode:
            * 'full_cov`: `n x m`
            * `diag`: `n`
        """
        # Size checking
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)    # Add a last dimension, if necessary
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        # Reshape lengthscale
        d = x1.shape[-1]
        if self.lengthscale is None:
            lengthscale = x1.new_full(size=(d,), fill_value=d, dtype=x1.dtype)
        else:
            lengthscale = self.lengthscale

        # Type checking
        if isinstance(lengthscale, (int, float)):
            lengthscale = x1.new_full(size=(d,), fill_value=lengthscale, dtype=x1.dtype)  # [d,] size Tensor
        
        if isinstance(lengthscale, Tensor):
            if lengthscale.ndimension() == 0 or max(lengthscale.size()) == 1:
                lengthscale = x1.new_full(size=(d,), fill_value=lengthscale.item(), dtype=x1.dtype)
            if not max(lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")
        
        lengthscale = lengthscale.reshape(-1)

        adjustment = x1.mean(dim=-2, keepdim=True)  # [d,] size tensor
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.sum(torch.abs(x1_-x2_), dim=-1)
        else:
            distance = torch.cdist(x1_, x2_, p=1)

        res = torch.exp(-distance).clamp_min(1e-15)
        return res


class LaplaceAdditiveKernel(nn.Module):
    """
    Computes a covariance matrix based on the Laplace additive kernel.

    :param lengthscale: Set this if you want a customized lengthscale. (Default: 1.0.)
    :type lengthscale: float, optional
    """
    def __init__(self, lengthscale=None):
        super().__init__()
        self.lengthscale = lengthscale

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None, 
                diag: bool = False, **params) -> Tensor:
        """
        :param x1: First set of data of shape :math:`(n,d)`.
        :type x1: torch.Tensor.float
        :param x2: Second set of data of shape :math:`(m,d)`.
        :type x2: torch.Tensor.float
        :param diag: Compute diagonal covariance matrix if `True`. It must be the case that `x1 == x2`.
        :type diag: bool, optional

        :return: The kernel matrix or vector. The shape depends on the kernel's mode:
            * 'full_cov`: `n x m`
            * `diag`: `n`
        """        
        # Size checking
        if x1.ndimension() == 1:
            x1 = x1.unsqueeze(1)    # Add a last dimension, if necessary
        if x2 is not None:
            if x2.ndimension() == 1:
                x2 = x2.unsqueeze(1)
            if not x1.size(-1) == x2.size(-1):
                raise RuntimeError("x1 and x2 must have the same number of dimensions!")
        else:
            x2 = x1

        # Reshape lengthscale
        d = x1.shape[-1]
        if self.lengthscale is None:
            lengthscale = x1.new_full(size=(d,), fill_value=d, dtype=x1.dtype)
        else:
            lengthscale = self.lengthscale

        # Type checking
        if isinstance(lengthscale, (int, float)):
            lengthscale = x1.new_full(size=(d,), fill_value=lengthscale, dtype=x1.dtype)  # tensor of size [d,]
        
        if isinstance(lengthscale, Tensor):
            if lengthscale.ndimension() == 0 or max(lengthscale.size()) == 1:
                lengthscale = x1.new_full(size=(d,), fill_value=lengthscale.item(), dtype=x1.dtype)
            if not max(lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")
        
        lengthscale = lengthscale.reshape(-1)

        adjustment = x1.mean(dim=-2, keepdim=True)  # tensor of size [d,]
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = torch.abs(x1_-x2_)
        else:
            distance = x1_.unsqueeze(dim=-2).repeat(*x1_.shape[:-2],1,x2_.shape[-2], 1) - x2_.unsqueeze(dim=-3).repeat(*x2_.shape[:-2],x1_.shape[-2], 1, 1)

        res = torch.sum(torch.exp(-distance), dim=-1)
        return res