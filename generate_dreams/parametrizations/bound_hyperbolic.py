import torch
from utils import rescale

class bound_hyperbolic_param():
    """
    Optimizes image in hyperbolic tangent space. Limits dream perturbation to given eps.
    Note: Returns list of params for optimizer per class. Not just params for optimizer
    """
    def __init__(self, x:torch.Tensor, device, eps:1.):
        self.epsilon = float(1e-8)
        
        self.down_range = torch.clamp(x, min=0, max=eps)
        self.up_range = torch.clamp(torch.add(torch.neg(x), 1), max=eps)
        # self.up_range = torch.clamp(torch.add(torch.neg(x), 1), min=0, max=eps)

        self.x = rescale(x, range=(-1+self.epsilon, 1-self.epsilon)).atanh().to(device)
        self.w = torch.zeros_like(self.x, device=device, requires_grad=True)

    def __call__(self):
        return [self.w], lambda: self.to_rgb_img(self.x, self.w)

    def to_rgb_img(self, x: torch.Tensor, w: torch.Tensor):
        _x = x.tanh()
        _w = self.scaled_weights(w)
        return torch.div(torch.add(torch.add(_x, _w), (1 + self.epsilon)), 2)

    def scaled_weights(self, weights: torch.Tensor):
        _weight = weights.tanh()
        # Multiply by 2 due to range. Original range 0,1 is now -1, 1 when tanh'ed
        scale_tens = torch.mul(torch.where(_weight > 0, self.up_range, self.down_range), 2)
        return torch.mul(_weight, scale_tens)