import torch
import torch.nn.functional as F

from typing import Optional
from torch import Tensor
from torch.distributions import Categorical
from gpytorch.likelihoods import SoftmaxLikelihood, DirichletClassificationLikelihood

class SmoothedCategorical(Categorical):
    def __init__(self, probs: Optional[Tensor]=None, logits: Optional[Tensor]=None, label_smoothing: float=0.0):
        super().__init__(probs=probs, logits=logits)
        self.smoothing = label_smoothing
        self.confidence = 1 - label_smoothing
        
    def log_prob(self, value: Tensor) -> Tensor:
        if value.ndim == 1:
            value = F.one_hot(value, self._num_events)
        if self.smoothing:
            value = value * self.confidence + self.smoothing / self._num_events
        value, log_pmf = torch.broadcast_tensors(value, self.logits)
        return (value * log_pmf).sum(-1)

class SmoothedSoftmaxLikelihood(SoftmaxLikelihood):
    def __init__(self, num_classes: Optional[int]=None, label_smoothing: float=0.0):
        super(SmoothedSoftmaxLikelihood, self).__init__(None, num_classes, None, None)
        self.label_smoothing = label_smoothing
    
    def forward(self, function_samples: Tensor, *params, **kwargs):
        num_data, _ = function_samples.shape[-2:]
        if num_data == self.num_features:
            function_samples = function_samples.transpose(-1, -2)
        return SmoothedCategorical(logits=function_samples, label_smoothing=self.label_smoothing)

class Dirichlet(DirichletClassificationLikelihood):
    def __init__(self, num_classes, alpha_epsilon=0.001, learn_additional_noise=True, num_samples=300, **kwargs):
        super().__init__(torch.tensor([0]), alpha_epsilon, learn_additional_noise)
        self.num_classes = num_classes
        self.num_samples = num_samples

    def prepare_targets(self, targets):
        targets = targets.long()
        alpha = self.alpha_epsilon * torch.ones(
            targets.shape[-1],
            self.num_classes,
            device=targets.device,
            dtype=targets.dtype
        )
        idx = torch.arange(len(targets))
        alpha[idx, targets] = alpha[idx, targets] + 1.0
        noise = torch.log(1 / alpha + 1.0)
        alpha = alpha.log() - 0.5 * noise
        return noise.t(), alpha.t()

    def expected_log_prob(self, targets, f):
        noise_covar, targets = self.prepare_targets(targets)
        self.noise_covar.noise.data = noise_covar
        return super().expected_log_prob(targets, f)
    
    def __call__(self, function_dist):
        function_samples = function_dist.rsample(torch.Size([self.num_samples])).transpose(-1, -2)
        function_dist = torch.distributions.Categorical(logits=function_samples)
        return function_dist
