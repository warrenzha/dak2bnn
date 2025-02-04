import torch
import gpytorch

from torch import Tensor
from gpytorch.means import Mean
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.distributions import MultivariateNormal

from ..nn import Variational
from ..variational import AmortizedVariationalStrategy, AmortizedMultitaskVariationalStrategy
from ..variational import AmortizedVariationalDistribution, AmortizedMeanFieldVariationalDistribution

class AmortizedSVGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        mean_module: Mean,
        covar_module: Kernel,
        num_inducing: int,
        num_tasks: int,
        mean_field: bool=False
    ):
        if mean_field:
            variational_distribution = AmortizedMeanFieldVariationalDistribution(num_inducing, torch.Size([num_tasks]))
        else:
            variational_distribution = AmortizedVariationalDistribution(num_inducing, torch.Size([num_tasks]))
        variational_strategy = AmortizedVariationalStrategy(self, variational_distribution)

        if num_tasks > 1:
            variational_strategy = AmortizedMultitaskVariationalStrategy(variational_strategy, num_tasks)

        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module
        self.num_inducing = num_inducing
        self.num_tasks = num_tasks

    def set_variational_parameters(self, Z: Tensor, m: Tensor, L: Tensor):
        if self.num_tasks > 1:
            self.variational_strategy.base_variational_strategy.inducing_points = Z
            self.variational_strategy.base_variational_strategy._variational_distribution.set_parameters(m, L)
        else:
            self.variational_strategy.inducing_points = Z
            self.variational_strategy._variational_distribution.set_parameters(m, L)

    def forward(self, x: Tensor) -> MultivariateNormal:
        function_mean = self.mean_module(x)
        function_covar = self.covar_module(x)
        return MultivariateNormal(function_mean, function_covar)

class IDSGP(gpytorch.Module):
    def __init__(self,
        feature_extractor: torch.nn.Module,
        variational_estimator: Variational,
        gplayer: AmortizedSVGP,
        likelihood: Likelihood
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.variational_estimator = variational_estimator
        self.gplayer = gplayer
        self.likelihood = likelihood

    def forward(self, x: Tensor):
        out = self.feature_extractor(x)
        Z, m, L = self.variational_estimator(out)
        self.gplayer.set_variational_parameters(Z, m, L)
        function_dist = self.gplayer(x.unsqueeze(-2))
        return function_dist
    
    @torch.inference_mode()
    def predict(self, x: Tensor, num_samples: int=500):
        self.eval()
        with gpytorch.settings.num_likelihood_samples(num_samples):
            targets_dist = self.likelihood(self(x))
        return targets_dist

class AVDKL(IDSGP):
    def forward(self, x: Tensor):
        out = self.feature_extractor(x)
        Z, m, L = self.variational_estimator(out)
        self.gplayer.set_variational_parameters(Z, m, L)
        function_dist = self.gplayer(out.unsqueeze(-2))
        return function_dist
