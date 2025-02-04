import torch
import gpytorch
import gpinfuser

class AVDKL(gpytorch.Module):
    def __init__(self, feature_extractor, num_inducing, saturation=torch.nn.Tanh(), likelihood=None, num_tasks=1):
        super(AVDKL, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = gpinfuser.models.AmortizedSVGP(
            mean_module = gpytorch.means.ConstantMean(),
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
            num_inducing = num_inducing,
            num_tasks = num_tasks,
        )
        self.num_inducing = num_inducing
        self.saturation = saturation
        self.likelihood = likelihood


    def forward(self, x):
        out = self.feature_extractor(x)
        variational_module = gpinfuser.nn.Variational(
            in_features=out.size(-1),
            num_tasks=1,
            num_inducing=self.num_inducing,
            num_features=self.feature_extractor.out_features,
            saturation=self.saturation,
        )
        Z, m, L = variational_module(out)
        self.gp_layer.set_variational_parameters(Z, m, L)
        res = self.gp_layer(out.unsqueeze(-2))
        return res