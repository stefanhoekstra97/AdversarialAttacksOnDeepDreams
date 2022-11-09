import torch


class linear_param():
    def __init__(self, tensor, device):
        if(tensor.dim() == 3):
            self.image_tensor = tensor.unsqueeze(0)
        elif(tensor.dim() == 4):
            self.image_tensor = tensor

        self.x = tensor.to(device).requires_grad_(True)

    def __call__(self):
        return [self.x], lambda: torch.clamp(self.x, min=0, max=1)