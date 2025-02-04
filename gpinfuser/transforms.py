import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch import Tensor
from typing import Tuple

class RandomMixup(nn.Module):
    def __init__(self, num_classes: int, p: float=0.5, alpha: float=1.0, inplace: bool=False):
        super(RandomMixup, self).__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace
        
    def forward(self, inputs: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.inplace:
            inputs = inputs.clone()
            target = target.clone()
            
        if target.ndim == 1:
            target = F.one_hot(target, self.num_classes).float()
            
        if torch.rand(1).item() >= self.p:
            return inputs, target
        
        inputs_rolled = inputs.roll(1, 0)
        target_rolled = target.roll(1, 0)
        
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        
        inputs_rolled.mul_(1.0 - lambda_param)
        inputs.mul_(lambda_param).add_(inputs_rolled)
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        
        return inputs, target

class RandomCutmix(nn.Module):
    def __init__(self, num_classes: int, p: float=0.5, alpha: float=1.0, inplace: bool=False):
        super(RandomCutmix, self).__init__()
        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace
        
    def forward(self, inputs: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if not self.inplace:
            inputs = inputs.clone()
            target = target.clone()
            
        if target.ndim == 1:
            target = F.one_hot(target, self.num_classes).float()
            
        if torch.rand(1).item() >= self.p:
            return inputs, target
        
        inputs_rolled = inputs.roll(1, 0)
        target_rolled = target.roll(1, 0)
        
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        
        _, H, W = T.functional.get_dimensions(inputs)
        
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))
        
        r = 0.5 * math.sqrt(1 - lambda_param)
        r_w = int(r * W)
        r_h = int(r * H)
        
        x1 = int(torch.clamp(r_x - r_w, min=0))
        y1 = int(torch.clamp(r_y - r_h, min=0))
        x2 = int(torch.clamp(r_x + r_w, max=W))
        y2 = int(torch.clamp(r_y + r_h, max=H))
        
        inputs[:, :, y1:y2, x1:x2] = inputs_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))
        
        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)
        
        return inputs, target