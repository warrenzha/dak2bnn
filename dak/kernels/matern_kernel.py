from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn as nn


class MaternKernel(nn.Module):
    """
    https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function

    :param nu: smoothness parameter, scalar
    :param lengthscale: lengthscale [d] size tensor
    """

    def __init__(self, nu=1.5, lengthscale=None) -> None:
        super().__init__()
        self.nu = nu
        self.lengthscale = lengthscale

    def forward(self, x1: Tensor, x2: Optional[Tensor] = None,
                diag: bool = False, **params):
        """
        :param x1: input, [n1, d] size tensor
        :param x2: input, [n2, d] size tensor
        :param diag: return diagonal only

        :return: matern(x1, x2), [n1, n2] size tensor
        """
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

        # Reshape lengthscale
        d = x1.shape[-1]
        if self.lengthscale is None:
            lengthscale = x1.new_full(size=(d,), fill_value=d, dtype=x1.dtype)
        else:
            lengthscale = self.lengthscale

        # Type checking
        if isinstance(lengthscale, (int, float)):
            lengthscale = x1.new_full(size=(d,), fill_value=lengthscale,
                                      dtype=x1.dtype)  # [d,] torch.Tensor([1., 1.,.., 1.])

        if isinstance(lengthscale, Tensor):
            if lengthscale.ndimension() == 0 or max(lengthscale.size()) == 1:
                lengthscale = x1.new_full(size=(d,), fill_value=lengthscale.item(), dtype=x1.dtype)
            if not max(lengthscale.size()) == d:
                raise RuntimeError("lengthscale and input must have the same dimension")

        lengthscale = lengthscale.reshape(-1)

        adjustment = x1.mean(dim=-2, keepdim=True)  # [d] size tensor
        x1_ = (x1 - adjustment).div(lengthscale)
        x2_ = (x2 - adjustment).div(lengthscale)
        x1_eq_x2 = torch.equal(x1_, x2_)

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                distance = torch.zeros(*x1_.shape[:-2], x1_.shape[-2], dtype=x1_.dtype, device=x1.device)
            else:
                distance = ((x1_ - x2_) ** 2).sum(dim=-1).sqrt()
        else:
            distance = torch.cdist(x1_, x2_, p=2)
            distance = distance.clamp_min(1e-15)

        if self.nu == 0.5:  # 1/2 Matern
            res = torch.exp(-distance)
        elif self.nu == 1.5:  # 3/2 Matern
            res = (1 + math.sqrt(3) * distance) * torch.exp(-math.sqrt(3) * distance)
        elif self.nu == 2.5:  # 5/2 Matern
            res = (1 + math.sqrt(5) * distance + 5 * distance ** 2 / 3) * torch.exp(-math.sqrt(5) * distance)
        else:
            p = int(self.nu - 0.5)
            sum = 0
            for i in range(p + 1):
                sum += math.factorial(p + i) / (math.factorial(i) * math.factorial(p - i)) \
                       * (2 * math.sqrt(2 * p + 1) * distance) ** (p - i)
            res = math.factorial(p) / math.factorial(2 * p) \
                  * sum * torch.exp(-math.sqrt(2 * p + 1) * distance)
        return res
