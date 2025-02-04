import torch
import torch.nn as nn
import gpytorch

from typing import List
from torch import Tensor
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.variational import (
    IndependentMultitaskVariationalStrategy,
    CholeskyVariationalDistribution,
    GridInterpolationVariationalStrategy
)
from gpytorch.utils.grid import ScaleToBounds
from gpytorch.constraints import Positive

# SVDKL
class GridInterpolationSVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        mean_module: Mean,
        covar_module: Kernel,
        num_inducing: int,
        num_tasks: int,
        grid_bounds: List=[-1, 1]
    ):
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = IndependentMultitaskVariationalStrategy(
            base_variational_strategy=GridInterpolationVariationalStrategy(
                model=self,
                grid_size=num_inducing,
                grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution
            ),
            num_tasks=num_tasks
        )

        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.num_tasks = num_tasks
        self.scaler = ScaleToBounds(*grid_bounds)

    def forward(self, x: Tensor) -> MultivariateNormal:
        function_mean = self.mean_module(x)
        function_covar = self.covar_module(x)
        return MultivariateNormal(function_mean, function_covar)

class DKL(gpytorch.Module):
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        gplayer: gpytorch.Module,
        likelihood: Likelihood
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gplayer = gplayer
        self.likelihood = likelihood

    def forward(self, x: torch.Tensor):
        out = self.feature_extractor(x)
        function_dist = self.gplayer(out)
        return function_dist

class SVDKL(DKL):
    def forward(self, x: torch.Tensor):
        out = self.feature_extractor(x)
        out = self.gplayer.scaler(out)
        out = out.transpose(-1, -2).unsqueeze(-1)
        function_dist = self.gplayer(out)
        return function_dist
    
    @torch.inference_mode()
    def predict(self, x: torch.Tensor, num_samples=500):
        self.eval()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            targets_dist = self.likelihood(self(x))
        return targets_dist

# DLVKL
class PCA(nn.Module):
    def __init__(self, num_components: int):
        super().__init__()
        self.num_components = num_components

    def forward(self, x):
        x_mean = x.mean(0)
        x_centered = x - x_mean
        x_covar = (x_centered.t() @ x_centered) / (x.size(0) - 1)
        
        eigenvalues, eigenvectors = torch.linalg.eigh(x_covar)
        topk_indices = torch.topk(eigenvalues, k=self.num_components).indices
        principal_components = eigenvectors[:, topk_indices]
        projected_data = x_centered @ principal_components

        return projected_data

class FeaturesVariationalDistribution(gpytorch.Module):
    def __init__(self, mlp: nn.Module, prior_noise: float=1.0, jitter: float=1e-3):
        super().__init__()
        
        self.jitter = jitter
        self.mlp = mlp
        self.latent_dim = mlp[-1].out_features // 2

        if mlp[0].in_features > self.latent_dim:
            self.pca = PCA(self.latent_dim)

        self.register_parameter('raw_prior_noise', nn.Parameter(torch.as_tensor(0.0)))
        self.register_constraint('raw_prior_noise', Positive())
        self.prior_noise = prior_noise

    @property
    def prior_noise(self):
        return self.raw_prior_noise_constraint.transform(self.raw_prior_noise)

    @prior_noise.setter
    def prior_noise(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_prior_noise)
        self.initialize(raw_prior_noise=self.raw_prior_noise_constraint.inverse_transform(value))

    def prior_distribution(self, x):
        dim = x.size(1)
        
        if dim > self.latent_dim:
            mean = self.pca(x)
        elif dim == self.latent_dim:
            mean = x.clone()
        else:
            mean = torch.cat([x, torch.zeros(len(x), self.latent_dim - x.size(1), device=x.device)], dim=1)
        variance = torch.ones_like(mean, device=x.device) * self.prior_noise
        prior_dist = MultivariateNormal(mean, torch.diag_embed(variance))

        return prior_dist

    def variational_distribution(self, x):
        mean, raw_variance = x[:, :self.latent_dim], x[:, self.latent_dim:]
        variance = torch.sigmoid(raw_variance) * self.prior_noise
        out = self.reparameterize(torch.randn(mean.size(), device=x.device), mean, variance)
        variational_dist = MultivariateNormal(mean, torch.diag_embed(variance))
        return out, variational_dist

    def reparameterize(self, x, mean, variance):
        return mean + x * (variance + self.jitter)

    def forward(self, x):
        out = self.mlp(x)
        out, variational_dist = self.variational_distribution(out)
        prior_dist = self.prior_distribution(x)
        return out, prior_dist, variational_dist

class DLVKL(gpytorch.Module):
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        gplayer: gpytorch.Module,
        likelihood: Likelihood,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.gplayer = gplayer
        self.likelihood = likelihood
    
    def forward(self, x):
        out, prior_dist_x, variational_dist_x = self.feature_extractor(x)
        function_dist = self.gplayer(out)
        return function_dist, prior_dist_x, variational_dist_x

    @torch.inference_mode()
    def predict(self, x: Tensor, num_samples: int=500):
        self.eval()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            targets_dist = self.likelihood(self(x)[0])
        return targets_dist