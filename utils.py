import torch

def rescale(tensor:torch.Tensor, range:tuple = (0, 1)):
    '''Rescales tensor to a set range.'''
    tensorMin = torch.min(tensor)
    tensorMax = torch.max(tensor)
  
    return tensor.add(-1*tensorMin).div(tensorMax - tensorMin).mul(range[1] - range[0]).add(range[0])
