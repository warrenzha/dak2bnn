import math
import torch
import torch.nn as nn
import gpytorch

import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="torch.sparse.SparseTensor")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


class KISSGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )

        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class SVDKL(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.), grid_size=64,
                 likelihood=None, embedding=None):
        super(SVDKL, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_dim = num_dim
        self.grid_bounds = grid_bounds
        self.grid_size = grid_size
        self.likelihood = likelihood
        self.embedding = embedding

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

        self.gp_layer = KISSGPLayer(num_dim=num_dim, grid_bounds=grid_bounds, grid_size=grid_size)

    def forward(self, x):
        features = self.feature_extractor(x)
        if self.embedding is not None:
            features = self.embedding(features)
        features = self.scale_to_bounds(features)
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)

        return res


#####################################
# Regression
#####################################
class AdditiveKISSGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim]),
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ),
            num_tasks=1,
            num_latents=num_dim,
            latent_dim=-1
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_dim]))

        self.base_covar_module = gpytorch.kernels.RBFKernel(
            lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp,
            ), batch_shape=torch.Size([num_dim])
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                self.base_covar_module,
                grid_size=grid_size,
                grid_bounds=[grid_bounds],
                num_dims=1
            ), batch_shape=torch.Size([num_dim])
        )

        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        # x is of size [grid_size, 1]
        mean_x = self.mean_module(x)  # [grid_size]
        covar_x = self.covar_module(x)
        out = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return out


class SVDKLRegression(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.), grid_size=64,
                 num_features=16, likelihood=None):
        super(SVDKLRegression, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_dim = num_dim
        self.grid_bounds = grid_bounds
        self.grid_size = grid_size
        self.num_features = num_features
        self.likelihood = likelihood

        self.embedding = nn.Linear(self.feature_extractor.out_features, num_features)
        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(self.grid_bounds[0], self.grid_bounds[1])

        self.gp_layer = AdditiveKISSGPLayer(num_dim=num_features, grid_bounds=grid_bounds, grid_size=grid_size)

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.embedding(features)
        features = self.scale_to_bounds(features)  # [batch_size, num_dim]
        # This next line makes it so that we learn a GP for each feature
        features = features.transpose(-1, -2).unsqueeze(-1)
        res = self.gp_layer(features)
        return res
