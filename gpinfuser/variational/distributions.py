import torch

from gpytorch.variational import _VariationalDistribution
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import (
    CholLinearOperator,
    TriangularLinearOperator,
    DiagLinearOperator
)

class AmortizedMeanFieldVariationalDistribution(_VariationalDistribution):
    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3):
        super().__init__(num_inducing_points, batch_shape, mean_init_std)
        self.inducing_points = torch.Tensor()
        self.variational_mean = torch.Tensor()
        self.variational_stddev = torch.Tensor()

    def set_parameters(self, variational_mean, variational_stddev):
        self.variational_mean = variational_mean
        self.variational_stddev = variational_stddev

    def forward(self):
        variational_covar = DiagLinearOperator(self.variational_stddev.pow(2))
        return MultivariateNormal(self.variational_mean, variational_covar)
    
    def initialize_variational_distribution(self, *args, **kwargs):
        pass

    def shape(self):
        return self.variational_mean.shape[-2:]

    @property
    def dtype(self):
        return self.variational_mean.dtype

    @property
    def device(self):
        return self.variational_mean.device

class AmortizedVariationalDistribution(_VariationalDistribution):
    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3):
        super().__init__(
            num_inducing_points=num_inducing_points,
            batch_shape=batch_shape,
            mean_init_std=mean_init_std
        )

        self.inducing_points = torch.Tensor()
        self.variational_mean = torch.Tensor()
        self.variational_covar = torch.Tensor()

    def forward(self):
        return MultivariateNormal(self.variational_mean, self.variational_covar)

    def initialize_variational_distribution(self, *args, **kwargs):
        pass

    def shape(self):
        return self.variational_mean.shape[-2:]

    @property
    def dtype(self):
        return self.variational_mean.dtype

    @property
    def device(self):
        return self.variational_mean.device

    def set_parameters(self, variational_mean, chol_variational_covar):
        self._set_variational_mean(variational_mean)
        self._set_chol_variational_covar(chol_variational_covar)

    def _set_variational_mean(self, variational_mean):
        self.variational_mean = variational_mean

    def _set_chol_variational_covar(self, chol_variational_covar):
        self.variational_covar = CholLinearOperator(
            TriangularLinearOperator(chol_variational_covar)
        )