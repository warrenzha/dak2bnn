import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'ReLU',
    'ReLUN',
    'MinMax',
    'ScaleToBounds',
]


class ReLU(nn.Module):
    """
    Implement ReLU activation used in BNN.
    
    :param inplace: can optionally do the operation in-place. It should be a [d] size tensor. (Default: `False`.)
    :type inplace: bool, optional
    """

    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        kl = 0
        return F.relu(x[0], inplace=self.inplace), kl

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class ReLUN(nn.Module):
    """
    Implement ReLU-N activation used in BNN.

    :param upper: Set this if you want a customized upper bound of ReLU. (Default: `1.0`.)
    :type upper: float, optional
    :param inplace: can optionally do the operation in-place. It should be a [d] size tensor. (Default: `False`.)
    :type inplace: bool, optional
    """

    __constants__ = ['inplace']

    def __init__(self, upper=1, inplace=False):
        super(ReLUN, self).__init__()
        self.inplace = inplace
        self.upper = upper

    def forward(self, x):
        ub = torch.ones_like(x, dtype=x.dtype, device=x.device) * self.upper
        x = torch.min(x, ub)
        return F.relu(x, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class MinMax(nn.Module):
    """
    Implement Min-Max normalization.

    :param lengthscale: Set this if you want a customized lengthscale. (Default: `1.0`.)
    :type lengthscale: float, optional
    :param bias: Set this if you want a customized bias. (Default: `None`.)
    :type bias: float, optional
    """

    def __init__(self, lengthscale=1., bias=0., eps=1e-05):
        super().__init__()
        self.lengthscale = lengthscale
        self.bias = bias
        self.eps = eps

    def forward(self, x):
        if (x.max() - x.min()) > self.eps:
            out = self.lengthscale * (x - x.min()) / (x.max() - x.min()) + self.bias
        else:
            out = x / x.max() + self.bias
        return out


class ScaleToBounds(nn.Module):
    """
    Scale the input data so that it lies in between the lower and upper bounds.

    :param float lower_bound: lower bound of scaled data
    :param float upper_bound: upper bound of scaled data
    """

    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.register_buffer("min_val", torch.tensor(lower_bound))
        self.register_buffer("max_val", torch.tensor(upper_bound))

    def forward(self, x):
        if self.training:
            min_val = x.min()
            max_val = x.max()
            self.min_val.data = min_val
            self.max_val.data = max_val
        else:
            min_val = self.min_val
            max_val = self.max_val
            # Clamp extreme values
            x = x.clamp(min_val, max_val)

        diff = max_val - min_val
        x = (x - min_val) * (0.95 * (self.upper_bound - self.lower_bound) / diff) + 0.95 * self.lower_bound
        return x


def minmax(x, lengthscale=1., bias=0., eps=1e-05):
    return MinMax(lengthscale, bias, eps)(x)


def relu(x, inplace=False):
    return ReLU(inplace)(x)


def relu_n(x, upper=6, inplace=False):
    return ReLUN(upper, inplace)(x)


def scale_to_bounds(x, lower_bound, upper_bound):
    return ScaleToBounds(lower_bound, upper_bound)(x)