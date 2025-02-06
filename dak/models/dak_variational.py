import torch
import gpytorch
import torch.nn as nn
from dak.layers import InducedPriorUnit, Amk1d, LinearReparameterization, LinearFlipout, Conv1dFlipout, NoiseLayer
from dak.layers.functional import ScaleToBounds
from dak.utils.sparse_design.design_class import HyperbolicCrossDesign
from dak.kernels.laplace_kernel import LaplaceProductKernel


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class DAK(nn.Module):
    def __init__(self, feature_extractor, num_classes=10, num_features=16, inducing_level=3, grid_bounds=(-10., 10.)):
        super(DAK, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.inducing_level = inducing_level
        self.grid_bounds = grid_bounds

        self.feature_extractor = feature_extractor
        self.embedding = nn.Linear(self.feature_extractor.out_features, num_features)
        self.scale_to_bounds = ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])
        self.gp_activation = Amk1d(
            in_features=num_features,
            n_level=inducing_level,
            input_lb=grid_bounds[0],
            input_ub=grid_bounds[1],
            design_class=HyperbolicCrossDesign,
            kernel=LaplaceProductKernel(lengthscale=1.),
        )
        self.gp_forward = LinearReparameterization(
            in_features=[self.gp_activation.in_features, self.gp_activation.num_inducing],
            out_features=num_classes,
            prior_mean=0.,
            prior_variance=1.,
            posterior_mu_init=0.,
            posterior_rho_init=-3.,
        )
        self.noise_layer = NoiseLayer(init_rho_sigma=-2.0)

    def forward(self, x, num_mc=1, return_kl=True, return_sampling=True, add_noise=False):
        """
        If return_sampling is True, res is tensor
        If return_sampling is False, res is a named tuple of mean and variance, and
            we can get mean by res.mean
            and can get variance by res.var
        """
        features = self.feature_extractor(x)
        features = self.embedding(features)
        features = self.scale_to_bounds(features)
        features = self.gp_activation(features)  # [N, P, M] size tensor --> [N, P*M] size tensor

        if return_sampling:
            output_ = []
            kl_ = []
            for mc_run in range(num_mc):
                output, kl = self.gp_forward(features, return_kl=return_kl, return_sampling=True)
                output_.append(output)
                kl_.append(kl)
            res = torch.stack(output_)
            kl = torch.stack(kl_)
        else:
            res, kl = self.gp_forward(features, return_kl=return_kl, return_sampling=False)

        if add_noise:
            res = self.noise_layer(res, return_sampling=return_sampling)

        if return_kl:
            return res, kl
        else:
            return res


class DAKMC(nn.Module):
    """DAK for Classification with Monte Carlo Sampling"""

    def __init__(self, feature_extractor,
                 num_features=64, num_tasks=10,
                 inducing_level=3, grid_bounds=(-1., 1.),
                 ):
        super(DAKMC, self).__init__()
        self.feature_extractor = feature_extractor
        self.embedding = nn.Linear(feature_extractor.fc.in_features, num_features, bias=False)

        self.scale_to_bounds = ScaleToBounds(grid_bounds[0], grid_bounds[1])

        self.gp_activation = InducedPriorUnit(
            in_features=num_features,
            induced_level=inducing_level,
            kernel=LaplaceProductKernel(lengthscale=1.),  # Choose the general kernel you want to use
            grid_bounds=grid_bounds,
        )
        self.gp_forward = LinearFlipout(
            in_features=self.gp_activation.out_features,
            out_features=num_tasks,
            bias=False,
        )

        self._init_params()

    def _init_params(self):
        self.apply(_weights_init)

    def forward(self, x, num_mc=1, return_kl=True):
        features = self.feature_extractor(x)
        if self.embedding is not None:
            features = self.embedding(features)
        features = self.scale_to_bounds(features)
        features = self.gp_activation(features).flatten(start_dim=1)

        output_ = []
        kl_ = []
        for mc_run in range(num_mc):
            output, kl = self.gp_forward(features, return_kl=return_kl)
            output_.append(output)
            kl_.append(kl)

        res = torch.mean(torch.stack(output_), dim=0)
        kl = torch.mean(torch.stack(kl_), dim=0)

        if return_kl:
            return res, kl
        else:
            return res


class MultitaskDAK(nn.Module):
    """DAK for Multi-task regression/classification with Monte Carlo Sampling"""

    def __init__(self, feature_extractor,
                 num_features=64, num_tasks=10,
                 inducing_level=3, grid_bounds=(-1., 1.),
                 ):
        super(MultitaskDAK, self).__init__()
        self.feature_extractor = feature_extractor
        self.embedding = nn.Linear(feature_extractor.fc.in_features, num_features, bias=False)

        self.scale_to_bounds = ScaleToBounds(grid_bounds[0], grid_bounds[1])

        self.gp_activation = InducedPriorUnit(
            in_features=num_features,
            induced_level=inducing_level,
            kernel=gpytorch.kernels.MaternKernel(nu=2.5),  # Choose the general kernel you want to use
            grid_bounds=grid_bounds,
        )
        self.gp_forward = Conv1dFlipout(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=self.gp_activation.num_inducing,
            groups=num_features,
            bias=False,
        )
        self.linear = nn.Linear(num_features, num_tasks, bias=True)

        self._init_params()

    def _init_params(self):
        self.gp_activation.kernel.lengthscale = 1.
        self.gp_activation.kernel.lengthscale.requires_grad = False
        self.apply(_weights_init)

    def forward(self, x, num_mc=1, return_kl=True):
        features = self.feature_extractor(x)
        if self.embedding is not None:
            features = self.embedding(features)
        features = self.scale_to_bounds(features)
        features = self.gp_activation(features)

        output_ = []
        kl_ = []
        for mc_run in range(num_mc):
            output, kl = self.gp_forward(features, return_kl=return_kl)
            output_.append(output)
            kl_.append(kl)

        res = torch.mean(torch.stack(output_), dim=0).squeeze(-1)
        kl = torch.mean(torch.stack(kl_), dim=0)
        res = self.linear(res)

        if return_kl:
            return res, kl
        else:
            return res