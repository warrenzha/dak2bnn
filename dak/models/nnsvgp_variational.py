import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, IndependentMultitaskVariationalStrategy
from gpytorch.variational import VariationalStrategy


class SVGPLayer(ApproximateGP):
    """
    Refer to
    https://docs.gpytorch.ai/en/stable/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html
    """

    def __init__(self, inducing_points, classification=False):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        if classification:
            variational_strategy = IndependentMultitaskVariationalStrategy(
                variational_strategy,
                num_tasks=1,
            )
        super(SVGPLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NNSVGP(gpytorch.Module):
    def __init__(self, feature_extractor, inducing_points, likelihood=None, classification=False):
        super(NNSVGP, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = SVGPLayer(inducing_points=inducing_points, classification=classification)
        self.inducing_points = inducing_points
        self.likelihood = likelihood
        self.classification = classification

    def forward(self, x):
        features = self.feature_extractor(x)

        if self.classification:
            features = features.unsqueeze(0)

        res = self.gp_layer(features)
        return res


class MultitaskGPLayer(gpytorch.models.ApproximateGP):
    def __init__(self, output_dims, num_inducing, input_dims=1):
        # Let's use a different set of inducing points for each latent function
        inducing_points = torch.rand(output_dims, num_inducing, input_dims)

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing, batch_shape=torch.Size([output_dims])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=output_dims,
        )

        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([output_dims]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([output_dims])),
            batch_shape=torch.Size([output_dims])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class NNSVGPClassification(gpytorch.Module):
    def __init__(self, feature_extractor, num_inducing, num_classes, likelihood=None, num_dim=1):
        super(NNSVGPClassification, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = MultitaskGPLayer(output_dims=num_classes,
                                         num_inducing=num_inducing,
                                         input_dims=num_dim)
        self.likelihood = likelihood

    def forward(self, x):
        features = self.feature_extractor(x)
        res = self.gp_layer(features)
        return res
