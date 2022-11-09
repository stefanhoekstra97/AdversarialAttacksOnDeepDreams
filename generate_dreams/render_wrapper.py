import torch
from torch import nn


# Custom Lucent (wrap/edited):
from generate_dreams.render_engine import generate_dream



class render_wrapper():
    def __init__(self, dreamy_model: nn.Module, class_list:list, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = dreamy_model.to(self.device)
        self.class_labels = class_list

    # model:torch.nn.Module,
    # batch,
    # opt_lr=1e-3,
    # iterations=(8,),
    # parametrization="tanh",
    # device = "cuda",
    # penal_f = None,
    # penal_factor=0.0,
    # **kwargs (eps)


    def set_dream_params(self, iterations: tuple = (32,), opt_lr: float = 1e-4, **kwargs):
        """Initialize dream params to the input variables as dictionary.
        args: 
        iterations: Tuple containing number of iterations we will optimize the image for
        opt_lr: Learning rate for the Adam optimizer
        opt_wd: Weight decay for optimizer
        """
        
        self.dream_params = {}
        self.dream_params["iterations"] = iterations
        self.dream_params["opt_lr"] = opt_lr

        self.dream_params["parametrization"] = "tanh"
        self.dream_params["penal_factor"] = 0
        self.dream_params["penal_f"] = None
        self.dream_params["limit_eps"] = 0

        for k,v in kwargs.items():
            print(f"Set {k}\n")
            self.dream_params[k] = v
        

    def generate_dreams(self, batch):
        dreams = generate_dream(self.model, batch=batch, device=self.device,
            opt_lr=self.dream_params["opt_lr"],
            iterations=self.dream_params["iterations"],
            parametrization=self.dream_params["parametrization"],
            penal_f=self.dream_params["penal_f"],
            penal_factor=self.dream_params["penal_factor"],
            limit_eps=self.dream_params["limit_eps"]             
            )

        return dreams[-1]

