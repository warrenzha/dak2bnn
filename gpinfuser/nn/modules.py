import torch
import torch.nn as nn

from torch import Tensor
from typing import Tuple, List, Callable, Optional, Union

class MLP(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        hidden_dim: List[int],
        nonlinearity: Callable[..., nn.Module]=nn.GELU(),
        dropout: float=0.0,
        norm_layer: Callable[..., nn.Module]=nn.BatchNorm1d
    ):
        layers = []
        for out_features in hidden_dim[:-1]:
            layers.append(nn.Linear(in_features, out_features))
            if norm_layer is not None:
                layers.append(norm_layer(out_features))
            layers.append(nonlinearity)
            if dropout:
                layers.append(nn.Dropout(dropout))
            in_features = out_features
        layers.append(nn.Linear(in_features, hidden_dim[-1]))

        super().__init__(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

class VariationalMeanField(nn.Module):
    def __init__(
            self,
            in_features: int,
            num_tasks: int,
            num_features: int,
            num_inducing: int,
            saturation: Callable[..., nn.Module]=nn.Sigmoid(),
            transform: Callable[..., nn.Module]=nn.Softplus()
        ):

        super().__init__()

        self.num_tasks = num_tasks
        self.num_features = num_features
        self.num_inducing = num_inducing
        self.saturation = saturation
        self.transform = transform

        num_inducing_units = num_features * num_inducing
        num_mean_units = num_tasks * num_inducing

        self.inducing_points = nn.Linear(in_features, num_inducing_units)
        self.variational_mean = nn.Linear(in_features, num_mean_units)
        self.variational_stddev = nn.Linear(in_features, num_mean_units)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        num_data = x.size(0)

        x = self.saturation(x)
        inducing_points = self.inducing_points(x)\
            .reshape((num_data, self.num_inducing, self.num_features))
        variational_mean = self.variational_mean(x)\
            .reshape((num_data, self.num_tasks, self.num_inducing))
        variational_stddev = self.variational_stddev(x)\
            .reshape((num_data, self.num_tasks, self.num_inducing))
        variational_stddev = self.transform(variational_stddev)
        
        return inducing_points, variational_mean.transpose(0, 1), variational_stddev.transpose(0, 1)
    
class Variational(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_tasks: int,
        num_features: int,
        num_inducing: int,
        saturation: Callable[..., nn.Module]=nn.Tanh(),
        transform_chol_diag: Optional[Union[nn.Module, Callable]]=nn.Softplus(),
        transform_chol_lower: Optional[Union[nn.Module, Callable]]=None
    ):
        super().__init__()

        self.num_tasks = num_tasks
        self.num_features = num_features
        self.num_inducing = num_inducing
        self.saturation = saturation
        self.transform_chol_diag = transform_chol_diag
        self.transform_chol_lower = transform_chol_lower

        self.chol_lower_dim = num_inducing * (num_inducing - 1) // 2
        self._chol_diag = torch.arange(num_inducing)
        self._chol_lower = torch.tril_indices(num_inducing, num_inducing, offset=-1)

        num_inducing_units = num_features * num_inducing
        num_mean_units = num_tasks * num_inducing
        num_chol_units = self.chol_lower_dim * num_tasks

        self.initialize_projections(
            in_features=in_features,
            num_inducing_units=num_inducing_units,
            num_mean_units=num_mean_units,
            num_chol_units=num_chol_units
        )
        self.reset_parameters()

    def initialize_projections(
        self,
        in_features: int,
        num_inducing_units: int,
        num_mean_units: int,
        num_chol_units: int
    ):
        self.induc_proj = nn.Linear(in_features, num_inducing_units)
        self.mean_proj = nn.Linear(in_features, num_mean_units)
        self.chol_diag_proj = nn.Linear(in_features, num_mean_units)
        self.chol_lower_proj = nn.Linear(in_features, num_chol_units)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def build_chol_variational_covar(self, diag: Tensor, lower: Tensor):
        if self.transform_chol_diag is not None:
            diag = self.transform_chol_diag(diag)

        if self.transform_chol_lower is not None:
            lower = self.transform_chol_lower(lower)

        shape = (diag.size(0), self.num_tasks, self.num_inducing, self.num_inducing)
        chol_var_covar = torch.zeros(shape, dtype=diag.dtype, device=diag.device)
        chol_var_covar[..., self._chol_diag, self._chol_diag] = diag[..., :]
        chol_var_covar[..., self._chol_lower[0], self._chol_lower[1]] = lower[..., :]
        return chol_var_covar

    def compute_variational_parameters(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        num_data = x.size(0)

        x = self.saturation(x)
        inducing_points = self.induc_proj(x, **kwargs)\
            .reshape([num_data, self.num_inducing, self.num_features])
        variational_mean = self.mean_proj(x, **kwargs)\
            .reshape([num_data, self.num_tasks, self.num_inducing])
        chol_diag = self.chol_diag_proj(x, **kwargs)\
            .reshape([num_data, self.num_tasks, self.num_inducing])
        chol_lower = self.chol_lower_proj(x, **kwargs)\
            .reshape([num_data, self.num_tasks, self.chol_lower_dim])
        chol_var_covar = self.build_chol_variational_covar(chol_diag, chol_lower)

        return inducing_points, variational_mean.transpose(0, 1), chol_var_covar.transpose(0, 1)

    def forward(self, x: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        return self.compute_variational_parameters(x, **kwargs)

class FeaturesWithVariationalParameters(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_tasks: int,
        num_inducing: int,
        dropout: float=0,
        nonlinearity_x: Callable[..., nn.Module]=nn.ReLU(),
        nonlinearity_q: Callable[..., nn.Module]=nn.Tanh()
    ):
        super().__init__()

        self.variational_estimator = Variational(
            in_features=in_features,
            num_tasks=num_tasks,
            num_features=out_features,
            num_inducing=num_inducing,
            dropout=dropout,
            nonlinearity=nonlinearity_q
        )
        self.features_estimator = nn.Sequential(
            nonlinearity_x,
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features)
        )
    
    def forward(self, x: Tensor, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.features_estimator(x)
        inducing, mean, chol_covar = self.variational_estimator(x)
        return features, inducing, mean, chol_covar
