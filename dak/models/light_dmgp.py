# Copyright (c) 2024 Wenyuan Zhao, Haoyuan Chen
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# @authors: Wenyuan Zhao, Haoyuan Chen.
#
# ===============================================================================================


from __future__ import print_function
import torch
import gpytorch
import torch.nn as nn
import torch.nn.functional as F

from dak.layers import LinearReparameterization, LightWeightLinear
from dak.layers import AMK

__all__ = [
    'DAMGPmnist',
]

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class DAMGPmnist(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 design_class,
                 kernel,
                 option='additive'):
        super(DAMGPmnist, self).__init__()

        self.option = option

        w0 = 64
        self.fc0 = LightWeightLinear(
            in_features=input_dim,
            out_features=w0,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 1st layer of DGP: input:[n, input_dim] size tensor, output:[n, w1] size tensor
        #################################################################################
        # return [n, m1] size tensor for [n, input_dim] size input and [m1, input_dim] size sparse grid
        self.gp1 = AMK(in_features=w0, n_level=5, design_class=design_class, kernel=kernel)
        m1 = self.gp1.out_features # m1 = input_dim*(2^n_level-1)
        w1 = 64
        # return [n, w1] size tensor for [n, m1] size input and [m1, w1] size weights
        self.fc1 = LightWeightLinear(
            in_features=m1,
            out_features=w1,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        #################################################################################
        ## 2nd layer of DGP: input:[n, w1] size tensor, output:[n, w2] size tensor
        #################################################################################
        # return [n, m2] size tensor for [n, w1] size input and [m2, w1] size sparse grid
        self.gp2 = AMK(in_features=w1, n_level=5, design_class=design_class, kernel=kernel)
        m2 = self.gp2.out_features # m2 = w1*(2^n_level-1)
        w2 = output_dim
        # return [n, w2] size tensor for [n, m2] size input and [m2, w2] size weights
        self.fc2 = LightWeightLinear(
            in_features=m2,
            out_features=w2,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
            bias=True,
        )

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-2, 2)

    def forward(self, x, num_mc=10):
        kl_sum = 0

        x = torch.flatten(x, 1)
        x, kl = self.fc0(x, num_mc=num_mc)
        kl_sum += kl

        x = self.scale_to_bounds(x)
        x = self.gp1(x)
        x, kl = self.fc1(x, num_mc=num_mc, lightweight=False)
        kl_sum += kl

        x = self.scale_to_bounds(x)
        x = self.gp2(x)
        x, kl = self.fc2(x, num_mc=num_mc, lightweight=False)
        kl_sum += kl

        output = F.log_softmax(x, dim=1)
        return output, kl_sum