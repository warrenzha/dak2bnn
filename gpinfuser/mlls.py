import torch
import gpytorch

from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import Likelihood

class LatentVariableVariationalELBO(VariationalELBO):
    def __init__(
        self,
        likelihood: Likelihood,
        model: gpytorch.Module,
        num_data: int,
        beta: float=1.0,
        inducing_beta: float=1.0,
        combine_terms: bool=True
    ):
        super().__init__(likelihood, model, num_data, inducing_beta, False)
        self.latent_variable_beta = beta
        self.combine_latent_variable_terms = combine_terms

    def forward(self, approximate_dist_f, prior_dist_x, approximate_dist_x, target, **kwargs):
        ll, kl, _ = super().forward(approximate_dist_f, target, **kwargs)
        kl_x = torch.distributions.kl.kl_divergence(approximate_dist_x, prior_dist_x)
        kl_x = kl_x.mean() * self.latent_variable_beta

        if self.combine_latent_variable_terms:
            return (ll - kl_x) - kl

        return ll, kl, kl_x