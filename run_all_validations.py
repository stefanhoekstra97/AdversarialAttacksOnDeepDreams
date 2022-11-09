# Global library imports
import time
import io
import itertools
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pytorch_lightning.callbacks import ModelCheckpoint

from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import make_grid

import torchmetrics

import foolbox as fb

import matplotlib

# Solve matplotlib mem issue when plotting imgs:
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import PIL

# Custom code imports
from lightning_trainer import LitModelTrainer
from utils import rescale

# Models to get validation results of.
### Identifier: model location
model_list = {
    # "base_model": "trained_models/MNIST/base_training/version_0/checkpoints/model-epoch=20-val_acc=0.994.ckpt",

    "base_adversarial_v0": "trained_models/MNIST/CW20-0.1/eps=4.2/base_adversarial/version_0/checkpoints/model-epoch=14-val_acc=0.994.ckpt",
    "base_adversarial_v1": "trained_models/MNIST/CW20-0.1/eps=4.2/base_adversarial/version_1/checkpoints/model-epoch=30-val_acc=0.994.ckpt",
    "base_adversarial_v2": "trained_models/MNIST/CW20-0.1/eps=4.2/base_adversarial/version_2/checkpoints/model-epoch=26-val_acc=0.994.ckpt",

    "dream4itslr1e-1_v0": "trained_models/MNIST/CW20-0.1/eps=4.2/dream4itslr1e-1/version_0/checkpoints/model-epoch=48-val_acc=0.991.ckpt",
    "dream4itslr1e-1_v1": "trained_models/MNIST/CW20-0.1/eps=4.2/dream4itslr1e-1/version_1/checkpoints/model-epoch=49-val_acc=0.991.ckpt",
    "dream4itslr1e-1_v2": "trained_models/MNIST/CW20-0.1/eps=4.2/dream4itslr1e-1/version_2/checkpoints/model-epoch=48-val_acc=0.992.ckpt",

    "dream4itslr1e-2_v0": "trained_models/MNIST/CW20-0.1/eps=4.2/dream4itslr1e-2/version_0/checkpoints/model-epoch=45-val_acc=0.994.ckpt",
    "dream4itslr1e-2_v1": "trained_models/MNIST/CW20-0.1/eps=4.2/dream4itslr1e-2/version_1/checkpoints/model-epoch=30-val_acc=0.995.ckpt",
    "dream4itslr1e-2_v2": "trained_models/MNIST/CW20-0.1/eps=4.2/dream4itslr1e-2/version_2/checkpoints/model-epoch=30-val_acc=0.994.ckpt",
    
    "dream8itslr1e-1_v0": "trained_models/MNIST/CW20-0.1/eps=4.2/dream8itslr1e-1/version_0/checkpoints/model-epoch=45-val_acc=0.975.ckpt",
    "dream8itslr1e-1_v1": "trained_models/MNIST/CW20-0.1/eps=4.2/dream8itslr1e-1/version_1/checkpoints/model-epoch=45-val_acc=0.977.ckpt",
    "dream8itslr1e-1_v2": "trained_models/MNIST/CW20-0.1/eps=4.2/dream8itslr1e-1/version_2/checkpoints/model-epoch=49-val_acc=0.984.ckpt",
    
    "dream8itslr1e-2_v0": "trained_models/MNIST/CW20-0.1/eps=4.2/dream8itslr1e-2/version_0/checkpoints/model-epoch=30-val_acc=0.993.ckpt",
    "dream8itslr1e-2_v1": "trained_models/MNIST/CW20-0.1/eps=4.2/dream8itslr1e-2/version_1/checkpoints/model-epoch=32-val_acc=0.994.ckpt",
    "dream8itslr1e-2_v2": "trained_models/MNIST/CW20-0.1/eps=4.2/dream8itslr1e-2/version_2/checkpoints/model-epoch=42-val_acc=0.995.ckpt"
    
}

model_bounds = (0, 1)
# FOR TESTING: Uses smaller subset of dataloader data.
#If set with x > 0,  First x images are validated on.
run_on_subset_size = 0

# Folder to save model results in, under validation_results
save_dir = 'MNIST/CW20-0.1/'


# File/folderpath (which exists) to save timing. Default (if empty) is validation_results.
timing_save = save_dir

# Base data location variables:
# cifar_loc = '../../data/cifar'
mnist_loc = '../../data/mnist'

classes_list = list(range(10)) # Just numbers for MNIST
# classes_list = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]

# Device settings:
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Dataloader
batch_size = 1024    # Num images per batch
num_procs = 0       # Number of workers to fetch data

# 20 -> max eps of 6.
# 28 = L2 multiplier for 0.3 -> 8.4 on MNIST
# 16 = L2 multiplier for 0.3 -> ~ 1.66 for CIFAR (sqrt((32*32*3)*(0.03*0.03))) = ~1.62
L2_eps_multiplier = 16      # L2 norm requires a much larger epsilon value as "budget" to work with. CIFAR.

# epsilons = [0.3] # For TESTING
epsilons = [0.01, 0.03, 0.1, 0.2, 0.3, 0.4, 0.5] # For MNIST
# epsilons = [0.001, 0.005, 0.01, 0.03, 0.1, 0.2, 0.3] # For CIFAR10


val_attacks = {
    "FGSM-RS": fb.attacks.FGSM(random_start=True),
    "FGSM-NORS": fb.attacks.FGSM(random_start=False),
    # "FGSM_L2_randomstart": fb.attacks.L2FastGradientAttack(random_start=True),
    # "FGSM_L2_norandomstart": fb.attacks.L2FastGradientAttack(random_start=False),
    
    "PGD4-0.5": fb.attacks.LinfPGD(steps=4, rel_stepsize=1/2),
    "PGD5-0.4": fb.attacks.LinfPGD(steps=5, rel_stepsize=2/5),

    "PGD10-0.1": fb.attacks.LinfPGD(steps=10, rel_stepsize=1/10),
    "PGD10-0.2": fb.attacks.LinfPGD(steps=10, rel_stepsize=2/10),
    "PGD10-0.4": fb.attacks.LinfPGD(steps=10, rel_stepsize=2/5),

    "PGD20-0.1": fb.attacks.LinfPGD(steps=20, rel_stepsize=1/10),
    "PGD20-0.2": fb.attacks.LinfPGD(steps=20, rel_stepsize=1/5),

    # "PGD40_r1o30": fb.attacks.LinfPGD(steps=40, rel_stepsize=1/30),

    "CW_L2_lr=0.01_steps=200_bstep=5": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=200, stepsize=0.01, abort_early=True, binary_search_steps=5, initial_const=1e-3),
    "CW_L2_lr=0.05_steps=100_bstep=5": fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=100, stepsize=0.05, abort_early=True, binary_search_steps=5, initial_const=1e-3),
    "CW_L2_lr=0.1_steps=100_bstep=5":  fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=100, stepsize=0.1, abort_early=True, binary_search_steps=5, initial_const=1e-3)
}

# Criterion to use (targeted vs untargeted) for attacks
adv_crit = fb.criteria.Misclassification

# Metrics to calculate/use. In identifier: (metric, reduction method)
# Supported reductions: sum, mean, ?conf.matrix?, avgdiff - > Calculates average distance of items in tensors
metricsPerAttack = {
    "Loss" :  lambda : torchmetrics.MeanMetric().to(device=torch_device),
    "Average_perturbation" : lambda : torchmetrics.MeanMetric().to(device=torch_device),
    "Top_1_Accuracy" : lambda : torchmetrics.Accuracy(top_k=1).to(device=torch_device),
    "Top_3_Accuracy" : lambda : torchmetrics.Accuracy(top_k=3), 
    "Confusion_Matrix" : lambda : torchmetrics.ConfusionMatrix(num_classes=10)
}


# Transform for data:
_transform = transforms.Compose([
    transforms.ToTensor()
])


_all_val_data = MNIST(root=mnist_loc, train=False, download=True, transform=_transform)
# _all_val_data = CIFAR10(root=cifar_loc, train=False, download=True, transform=_transform)

if run_on_subset_size > 0:
    val_data = torch.utils.data.Subset(_all_val_data, list(range(run_on_subset_size)))
else: 
    val_data = _all_val_data


val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=False)



## Helping function: Makes plt figure of tensors, with labels above each img.
def gen_labelled_fig(img_tensors: torch.Tensor, img_labels: list, title:str, raw_img: torch.Tensor=None):
    # We get tensors in range (-1, 1), normalize then map to image range

    base_diff_tensor = img_tensors[0].repeat(img_tensors.shape[0], 1, 1, 1)
    plot_tensor = torch.cat([img_tensors, rescale(torch.sub(img_tensors, base_diff_tensor))])


    grid = make_grid(plot_tensor, padding=8, normalize=True, pad_value=1, nrow=img_tensors.shape[0])
    # grid = make_grid(plot_tensor, padding=4, normalize=True, pad_value=1, nrow=img_tensors.shape[0])
    grid = grid.detach().permute(dims=(1,2,0))
    
    fig = plt.figure()
    axs = fig.add_axes([0,0,1,1])
      
    axs.imshow(grid)
    axs.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axs.set_title(title)
    plt.axis('off')

    for i, lbl in enumerate(img_labels):
        # axs.text(x=((i*34 )+ 4), y=82, s=lbl.rstrip(".0"))
        axs.text(x=((i*36 )+ 8), y=82, s=lbl.rstrip(".0"))

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

## Helping function: Creates confusion matrix.
def plot_confusion_matrix(cm, eps = "", att_id = "", class_names = list(range(10)), top_1_acc=0.0):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    
    plt.text(0, -1.5, f'attack_id: {att_id}; eps:{eps.rstrip(".0")}; acc: {top_1_acc:.3f}',
        horizontalalignment='center',
        verticalalignment='center')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf




# Validation funct.
def validate_attacks(fb_model, val_model, model_criterion, logging_Obj):
    attack_num_handled = 0
    metric_val_dict = {}

    for attack_idx, attack_tuple  in enumerate(val_attacks.items()):
        timer_start_att = time.perf_counter()
        val_attack: fb.attacks.base
        attack_id, val_attack = attack_tuple
        if "_L2" in attack_id:
            epsilon_mult = L2_eps_multiplier
        else:
            epsilon_mult = 1

        print("model ", model_identifier, " performing val of " + attack_id + ", attack step# = ", attack_num_handled)
        metric_val_dict[attack_id] = {}
        metric_dict = {}
        save_imgs = torch.Tensor(size=(3,32,32))
        for step, batch in enumerate(val_loader):
            x:torch.Tensor
            y:torch.Tensor
            x, y = batch
            x, y = x.to(torch_device), y.to(torch_device)
            
            raw_advs, clipped_advs, success = val_attack(fb_model, x, epsilons=np.multiply(epsilons, epsilon_mult), criterion=adv_crit(y))

            for eps_idx, epsilon_val in enumerate(epsilons):

                # Setup if on first step of per epsilon
                if step == 0:
                    metric_dict[str(epsilon_val)] = {}
                    metric_val_dict[attack_id][str(epsilon_val)] = {}
                x_adv = clipped_advs[eps_idx].to(torch_device)

                with torch.no_grad():      # Do forward pass
                    y_hat_adv = val_model(x_adv)
                    y_hat_adv.to(torch_device)

                   
                    metric: torchmetrics.Metric
                    for metric_id, metric in metricsPerAttack.items():
                        if step == 0:
                            # metric_val_dict[attack_id][str(epsilon_val)][metric_id] = {}
                            metric_dict[str(epsilon_val)][metric_id] = metric()

                        if metric_id == "Average_perturbation": # Calculate avg perturbation size on all images
                            alldiffstensor = torch.sub(x, x_adv).to(torch_device)
                            allmin, allmax = torch.aminmax(dim=0, input=alldiffstensor)
                            allmin, allmax = allmin.to(torch_device), allmax.to(torch_device)
                            metric_dict[str(epsilon_val)][metric_id](torch.sub(allmax, allmin))

                        elif metric_id == "Loss":  # Calculate model loss with model criterion
                            lossval = model_criterion(y_hat_adv, y)
                            metric_dict[str(epsilon_val)][metric_id](lossval)
                        else:
                            metric_dict[str(epsilon_val)][metric_id](y_hat_adv, y)

                    #  We want to log the first image of the first batch for each attack, with it's details on perturbation etc.
                    if step == 0:
                        log_step = 10 * attack_idx + eps_idx

                        diff_tensor = torch.sub(x[0], x_adv[0])

                        min_val, max_val = torch.aminmax(diff_tensor)
                        perturb_size = max_val.item() - min_val.item()
                        logging_Obj.add_scalar(tag=f"pertsize_of_eps{epsilon_val:.3f}", scalar_value=perturb_size, global_step=log_step)
                        
                        if eps_idx == 0:
                            save_imgs = torch.unsqueeze(x[0].detach().clone(), dim=0)
                            save_raw_adv = torch.unsqueeze(raw_advs[eps_idx][0].detach().clone(), dim=0)

                        save_imgs = torch.cat([save_imgs, torch.unsqueeze(x_adv[0].detach().clone(), dim=0)], dim=0)
                        save_raw_adv = torch.cat([save_raw_adv, torch.unsqueeze(raw_advs[eps_idx][0].detach().clone(), dim=0)], dim=0)

        fig_buf = gen_labelled_fig(save_imgs.cpu(), ["Original"] + ["eps: " + str(f"{epsil:.3f}") for epsil in np.multiply(epsilons, epsilon_mult)], attack_id)

        _img = PIL.Image.open(fig_buf)
        _img = transforms.ToTensor()(_img)
        logging_Obj.add_image(tag="perturbed imgs", img_tensor=_img, global_step=attack_idx)
        
        # Compound metrics for this validation attack vector
        epsilon_num = 0
        for epsilon_val_str, metric_pair in metric_dict.items():
            metric_num = 0
            metric_f: torchmetrics.Metric
            for metric_id, metric_f in metric_pair.items():
                metric_step = 100 * attack_num_handled + 10 * epsilon_num + metric_num
                if metric_id == "Loss":
                    metric_value = metric_f.compute()
                    metric_val_dict[attack_id][epsilon_val_str][metric_id] = metric_value.item()
                    logging_Obj.add_scalar(tag=metric_id, scalar_value=metric_value, global_step=metric_step)
                elif metric_id == "Confusion_Matrix":
                    metric_value = metric_f.compute()
                   
                    cm_buffer = plot_confusion_matrix(cm=metric_value.detach().cpu().numpy(), eps=f"{(float(epsilon_val_str) * epsilon_mult):.3f}", att_id=attack_id, class_names=classes_list, top_1_acc=metric_val_dict[attack_id][epsilon_val_str]["Top_1_Accuracy"])

                    cm_img = PIL.Image.open(cm_buffer)
                    cm_img = transforms.ToTensor()(cm_img)
                    logging_Obj.add_image(tag=metric_id, img_tensor=cm_img, global_step=metric_step)
                    metric_val_dict[attack_id][epsilon_val_str][metric_id] = metric_value.tolist()
                    # metric_f.reset()
                else:
                    metric_value = metric_f.compute()
                    logging_Obj.add_scalar(tag=metric_id, scalar_value=metric_value, global_step=metric_step)
                    metric_val_dict[attack_id][epsilon_val_str][metric_id] = metric_value.item()
                    # metric_f.reset()
                metric_num += 1
            epsilon_num +=1
        attack_num_handled += 1
        timer_stop_attack = time.perf_counter()
        timing_dict[model_identifier + " - " + attack_id] = timer_stop_attack - timer_start_att
    return metric_val_dict



timing_dict = {}

for model_identifier, model_loc in model_list.items():
    # Load model from checkpoint:
    timer_start_model = time.perf_counter()
    mcp = LitModelTrainer.load_from_checkpoint(model_loc)
    model : nn.Module = mcp.model

    model = model.to(torch_device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Setup model in FB for attacks
    fmodel = fb.models.pytorch.PyTorchModel(model=model, bounds=model_bounds, device=torch_device)

    # Criterion to use for model loss calculation (not for training!)
    model_crit = nn.CrossEntropyLoss()

    # SummaryWriter for metric info and saved images:
    tf_writer = SummaryWriter(log_dir="validation_results/" + save_dir + model_identifier)

    dict_result = validate_attacks(fb_model=fmodel, val_model=model, model_criterion = model_crit, logging_Obj = tf_writer)
    timer_stop_model = time.perf_counter()
    timing_dict[model_identifier + " total"] = timer_stop_model - timer_start_model

    with open('validation_results/' + save_dir + model_identifier + '.txt', 'w') as f:
        json.dump(dict_result,f, indent=None)
        f.flush()
        f.close()

    with open('validation_results/' + timing_save + model_identifier + "timing.txt", 'w') as f:
        json.dump(timing_dict, f, indent=4)
        f.flush()
        f.close()
    timing_dict = {}

