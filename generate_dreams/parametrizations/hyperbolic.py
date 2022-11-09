import torch
from utils import rescale


class hyperbolic_param():
    """
    Optimizes image in hyperbolic tangent space, using a new variable set to zero initially.
    Can therefore be used with optimizers applying weight decay.
    Note: Returns list of params for optimizer per class. Not just params for optimizer
    """
    def __init__(self, x:torch.Tensor, device):
        self.epsilon = float(1e-8)
        
        # X in tanh space:
        self.x = rescale(x, range=(-1+self.epsilon, 1-self.epsilon)).atanh().to(device)
        self.w = torch.zeros_like(self.x, device=device, requires_grad=True)

    def __call__(self):
        return [self.w], lambda: self.to_rgb_img(torch.add(self.x, self.w))

    def to_rgb_img(self, tensor: torch.Tensor):
        return torch.div(torch.add(torch.tanh(tensor), (1 + self.epsilon)), 2)
