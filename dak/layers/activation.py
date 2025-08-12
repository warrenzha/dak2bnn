import torch
import gpytorch
import torch.nn as nn

from dak.kernels.laplace_kernel import LaplaceProductKernel
from dak.utils.sparse_design.design_class import HyperbolicCrossDesign
from dak.utils.operators.chol_inv import mk_chol_inv

__all__ = [
    'Amk1d',
    'Amk2d',
    'InducedPriorUnit',
]


class InducedPriorUnit(nn.Module):
    def __init__(self,
                 in_features,
                 induced_level,
                 kernel,
                 design_class=HyperbolicCrossDesign,
                 grid_bounds=(-2., 2.),
                 ):
        super(InducedPriorUnit, self).__init__()

        # induced grids U of dyadic sort design
        dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)
        # design_points of size M = 2^induced_level - 1
        design_points = dyadic_design(deg=induced_level, input_lb=grid_bounds[0], input_ub=grid_bounds[1]).points
        # add a last dimension if necessary
        if design_points.ndimension() == 1:
            design_points = design_points.reshape(-1, 1)

        self.in_features = in_features  # D
        self.num_inducing = design_points.shape[0]  # M
        self.out_features = self.num_inducing * in_features  # M * D

        # Cholesky decomposition of K_UU = L * L^T, compute {L^-1}^T
        covar = kernel(design_points)  # covariance matrix K of size (M, M)
        L_inv_T = gpytorch.root_inv_decomposition(covar, method='cholesky').to_dense().clone().detach()

        self.register_buffer('design_points', design_points)
        self.register_buffer('L_inv_T', L_inv_T)

        self.kernel = kernel

    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # reshape x of size (N, D) --> size (N, D, 1)
        x = self.kernel(x, self.design_points)  # k(x, U) of size (N, D, M)
        x = x @ self.L_inv_T  # k(x, U) * {L^-1}^T of size (N, D, M)
        return x


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


class AMK(nn.Module):
    r"""
    Implements additive markov GP as an activation layer using additive structure.

    .. math::

        \begin{equation*}
            \left\{ k\left( x_i, X^{SG} \right)R^{-1} \right\}^{d}_{i=1}
        \end{equation*}

    :param in_features: Size of each input sample.
    :type in_features: int
    :param n_level: Level of induced points for approximating GP. (Default: `3`.)
    :type n_level: int, optional
    :param input_lb: Input lower boundary. (Default: `-2`.)
    :type input_lb: float, optional
    :param input_ub: Input upper boundary. (Default: `2`.)
    :type input_ub: float, optional
    :param design_class: Base design class of sparse grid. (Default: `HyperbolicCrossDesign`.)
    :type design_class: class, dmgp.utils.sparse_design.design_class, optional
    :param kernel: Kernel function of deep GP. (Default: `LaplaceProductKernel(lengthscale=1.)`.)
    :type kernel: class, dmgp.kernels, optional
    """

    def __init__(self,
                 in_features,
                 n_level=3,
                 input_lb=-2,
                 input_ub=2,
                 kernel=LaplaceProductKernel(lengthscale=1.),
                 design_class=HyperbolicCrossDesign,
                 ):
        super().__init__()

        self.kernel = kernel

        dyadic_design = design_class(dyadic_sort=True, return_neighbors=True)(deg=n_level, input_lb=input_lb, input_ub=input_ub)
        chol_inv = mk_chol_inv(dyadic_design=dyadic_design, markov_kernel=kernel, upper=True)  # [m, m] size tensor
        design_points = dyadic_design.points.reshape(-1, 1)  # [m, 1] size tensor

        self.register_buffer('design_points',
                             design_points)  # [m,d] size tensor, sparse grid points X^{SG} of dyadic sort
        self.register_buffer('chol_inv',
                             chol_inv)  # [m,m] size tensor, inverse of Cholesky decomposition of kernel(X^{SG},X^{SG})
        self.out_features = design_points.shape[0] * in_features  # m*d

    def forward(self, x):
        r"""
        Computes the element-wise markov kernel activation of :math:`\mathbf x`.

        :param x: [N, C] size tensor, N is the batch size, C is the channels of input, L is the sequence length
        :type x: torch.Tensor.float

        :return: [N, C*L*M] size tensor, kernel(input, sparse_grid) @ chol_inv
        """

        out = torch.flatten(x, start_dim=1).unsqueeze(dim=-1)  # reshape x of size [N, C, L] --> size [N, C*L, 1]
        out = self.kernel(out, self.design_points)  # [N, C*L, M] size tenosr
        out = torch.matmul(out, self.chol_inv)  # [N, C*L, M] size tensor
        out = torch.flatten(out, start_dim=1)  # [N, C*L*M] size tensor

        return out
