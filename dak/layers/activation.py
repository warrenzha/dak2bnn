import torch
import torch.nn as nn

from dak.kernels.laplace_kernel import LaplaceProductKernel
from dak.utils.sparse_design.design_class import HyperbolicCrossDesign
from dak.utils.operators.chol_inv import mk_chol_inv

__all__ = [
    'Amk1d',
    'Amk2d',
]


class Amk1d(nn.Module):
    def __init__(self,
                 in_features,
                 n_level,
                 input_lb=-2,
                 input_ub=2,
                 design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        super().__init__()

        self.kernel = kernel

        dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_lb=input_lb,
                                                                              input_ub=input_ub)
        chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)  # [m, m] size tensor
        design_points = dyadic_design.points.reshape(-1, 1)  # [m, 1] size tensor

        self.register_buffer('design_points',
                             design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv',
                             chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})

        self.in_features = in_features  # D
        self.num_inducing = design_points.shape[0]
        self.out_features = self.num_inducing * in_features  # M * D

    def forward(self, x):
        out = x.unsqueeze(dim=-1)  # reshape x of size [N, D] --> size [N, D, 1]
        out = self.kernel(out, self.design_points)  # [N, D, M] size tensor
        out = torch.matmul(out, self.chol_inv)  # [N, D, M] size tensor

        return out


class Amk2d(nn.Module):
    def __init__(self,
                 in_channels,
                 n_level=3,
                 input_lb=-2,
                 input_ub=2,
                 design_class=HyperbolicCrossDesign,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 ):
        super(Amk2d, self).__init__()

        self.kernel = kernel

        dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_lb=input_lb,
                                                                              input_ub=input_ub)
        chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)  # [m, m] size tensor
        design_points = dyadic_design.points.reshape(-1, 1)  # [m, 1] size tensor

        self.register_buffer('design_points',
                             design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv',
                             chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_channels = design_points.shape[0] * in_channels

    def forward(self, x):
        pixel_size = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        num_pixels = x.shape[1]
        out = x.flatten(1).unsqueeze(dim=-1)
        out = self.kernel(out, self.design_points)  # [...,n*d, m] size tenosr
        out = torch.matmul(out, self.chol_inv)  # [..., n*d, m] size tensor
        out = out.reshape(-1, num_pixels, self.out_channels).transpose(1, 2)
        out = out.view(-1, self.out_channels, *pixel_size)

        return out
