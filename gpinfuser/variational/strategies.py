import torch

from torch import Tensor
from gpytorch.variational import VariationalStrategy, IndependentMultitaskVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator

class AmortizedVariationalStrategy(VariationalStrategy):
    def __init__(self, model, variational_distribution, jitter_val=1e-4):
        super().__init__(
            model, torch.Tensor(0), variational_distribution,
            jitter_val=jitter_val, learn_inducing_locations=False
        )

        delattr(self, 'inducing_points')
        self.inducing_points = torch.Tensor()
        self.variational_params_initialized.fill_(1)

    def kl_divergence(self):
        return super().kl_divergence().mean(-1)

    def forward(self, x: Tensor, *args, **kwargs):
        function_dist = super().forward(x, *args, **kwargs)
        function_mean = function_dist.mean.squeeze(-1)
        function_covar = DiagLinearOperator(
            function_dist.covariance_matrix\
                .squeeze(-1)\
                .squeeze(-1)
        )
        return MultivariateNormal(function_mean, function_covar)
    
    def __call__(self, x: Tensor, prior=False, **kwargs):
        self._clear_cache()
        return super().__call__(x, prior=prior, **kwargs)

class AmortizedMultitaskVariationalStrategy(IndependentMultitaskVariationalStrategy):
    def kl_divergence(self):
        return self.base_variational_strategy.kl_divergence().sum()
