import torch
import gpytorch

from torch import Tensor
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.variational import (
    VariationalStrategy,
    IndependentMultitaskVariationalStrategy,
    CholeskyVariationalDistribution
)

class SVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        mean_module: Mean,
        covar_module: Kernel,
        inducing_points: Tensor,
        num_tasks: int,
        learn_inducing_locations: bool=True
    ):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.shape[-2], batch_shape=torch.Size([num_tasks])
        )
        variational_strategy = VariationalStrategy(
            self, inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations
        )
        if num_tasks > 1:
            variational_strategy = IndependentMultitaskVariationalStrategy(variational_strategy, num_tasks)

        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.num_inducing = inducing_points.shape[-2]
        self.num_tasks = num_tasks

    def forward(self, x: Tensor) -> MultivariateNormal:
        function_mean = self.mean_module(x)
        function_covar = self.covar_module(x)
        return MultivariateNormal(function_mean, function_covar)
    
class GaussianProcess(gpytorch.Module):
    def __init__(self, gplayer: gpytorch.Module, likelihood: Likelihood):
        super().__init__()
        self.gplayer = gplayer
        self.likelihood = likelihood

    def forward(self, x: Tensor) -> MultivariateNormal:
        return self.gplayer(x)
    
    @torch.inference_mode()
    def predict(self, x: Tensor, num_samples: int=500):
        self.eval()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            targets_dist = self.likelihood(self(x))
        return targets_dist
