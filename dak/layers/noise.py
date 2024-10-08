import torch
import torch.nn as nn

from .linear import mean_var_tuple

__all__ = ['NoiseLayer']


class NoiseLayer(nn.Module):
    def __init__(self, init_rho_sigma=-2.0):
        """
        NoiseLayer adds i.i.d. Gaussian noise to the input.

        :param init_rho_sigma: Initial value of log(sigma), where sigma is the standard deviation of the noise.
        """
        super(NoiseLayer, self).__init__()
        # Learnable parameter for log(sigma), initialized to a small value
        self.rho_sigma = nn.Parameter(torch.tensor(init_rho_sigma))

    def forward(self, x, return_sampling=True):
        """
        Forward pass for the noise layer.
        Adds Gaussian noise N(0, sigma^2) where sigma is learnable.

        :param x: The input tensor
        :param return_sampling: If True, return the sampled output. If False, return the mean and variance.

        :return: The input tensor with added noise
        """
        # Compute sigma in a numerically stable way
        sigma = torch.log1p(torch.exp(self.rho_sigma))  # Ensure sigma is positive using log1p
        if return_sampling:
            noise = torch.randn_like(x) * sigma  # Generate Gaussian noise with std deviation sigma
            out = x + noise  # Add noise to the input
        else:
            out_mean, out_var = x.mean, x.var
            out_var += sigma ** 2
            out = mean_var_tuple(mean=out_mean, var=out_var)
        return out
