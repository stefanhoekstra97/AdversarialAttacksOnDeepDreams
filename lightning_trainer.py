# Library imports:
import torch
from torch import nn

import pytorch_lightning as pl

import torchmetrics

import foolbox as fb

#Custom imports:
from generate_dreams.render_wrapper import render_wrapper
from models.cifar_nn import Cifar_nn
from models.cifar_nn_nobn import Cifar_nn_nobn
from models.mnist_nn import Mnist_nn
from models.mnist_nn_nobn import Mnist_nn_nobn
from utils import rescale

# nn_archs: One of ["Fashion_mnist", "Mnist", "Cifar"]

class LitModelTrainer(pl.LightningModule):
    def __init__(self, with_adversarial: bool = False, adv_eps:float = 0.03, model: nn.Module = None, opt_lrs:float =1e-3, opt_wd=0.0, nn_arch: str = "Cifar", dream_gen_dict: dict = {}, attack = None, **kwargs):
        super().__init__()
        # Do optimization such as zero_grad and backward() ourselves
        self.automatic_optimization = False
        self.log_every_x = 19
        self.model_bounds = (0, 1)
        # # If no model is supplied, check architecture if given. Otherwise assume CIFAR.
        if(model is not None):
            self.model = model
        else:
            if(nn_arch == "Mnist"):
                self.model = Mnist_nn()
            elif(nn_arch == "Mnist-nobn"):
                self.model = Mnist_nn_nobn()
            elif nn_arch == "Cifar":
                self.model = Cifar_nn()
            elif nn_arch == "Cifar-nobn":
                self.model = Cifar_nn_nobn()
            else:
                raise Exception("No suitable architecture defined.")

        self.model.eval()

        self.img_log_step = 0
        self.val_img_log_step = 0

        #  Create metrics for evaluation and criterion.
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy_train = torchmetrics.Accuracy()
        self.accuracy_train_adv = torchmetrics.Accuracy()
        self.accuracy_val = torchmetrics.Accuracy()
        self.accuracy_val_adv = torchmetrics.Accuracy()
        self.accuracy_test = torchmetrics.Accuracy()

        self.opt_lrs = opt_lrs
        self.opt_wd = opt_wd
        self.with_adversarial = with_adversarial

        # Initialize foolbox model
        self.epsilons_adv = [adv_eps]

        self.fmodel = fb.models.pytorch.PyTorchModel(model=self.model, bounds=self.model_bounds)
       
        # Initialize dream generation if dream params are given, can be empty dict for no dream generation.
        self.dream_param_dict = dream_gen_dict
        if dream_gen_dict:
            self.dream_generator = render_wrapper(self.model, class_list=range(10), device='auto')
            self.dream_generator.set_dream_params(**self.dream_param_dict)

        if attack is None and with_adversarial:
            raise Exception("Attack not provided, but with adversarial is selected.")

        self.attack : fb.attacks.base = attack

        attackstr = self.attack.__str__() if self.attack is not None else "None"

        if "Projected" in attackstr:
            _attack_id = "PGD"
        elif "CarliniWagner" in attackstr:
            _attack_id = "CW"
        elif "FastGradient" in attackstr:
            _attack_id = "FGSM"
        else: 
            _attack_id = "None/Unknown"

        # Skeleton hyper parameters. Otherwise only joint parameters shown in TB.
        log_param_dict = {
                "nn_arch": nn_arch,
                "attack_id": _attack_id,
                "train_lr": opt_lrs, 
                "train_opt_wd":opt_wd,
                "train_adv_eps": adv_eps,

                "attack_abs_stepsize": -1,
                "attack_random_start": -1,
                "attack_rel_stepsize": -1,
                "attack_steps": -1,
                "attack_binary_search_steps": -1,
                "attack_stepsize": -1,
                "attack_initial_const":-1,
                "attack_confidence": -1,

                "dream_iterations": -1,
                "dream_opt_lr": -1
              }

        for k,v in dream_gen_dict.items():
            if "iterations" in k:
                log_param_dict["dream_" + k] = v[-1]
            else:
                log_param_dict["dream_" + k] = v

        if self.attack is not None:
            for k,v in self.attack.__dict__.items():
                if v:
                    log_param_dict["attack_" + k] = v

        self.save_hyperparameters(log_param_dict)
  

    def forward(self, x: torch.Tensor):
        prediction = self.model(x)
        return prediction

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        opt.zero_grad()
        x, y = batch

        # Disable gradient tracking on model:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval() # Start in eval mode for possibly dreams and/or adv. attack generation

        if(self.dream_param_dict):
            x_all_cl = x.detach().clone()
            x_clone = x[0].detach().clone()
            x_dream = self.generate_dream_batch(batch)

            if (batch_idx % self.log_every_x == 0):
                dreamimg = x_dream[0].detach().clone()

                baseimg = x_clone
                dreamscaled = rescale(dreamimg)
                basescaled = rescale(baseimg)

                diff_abs = torch.sub(dreamimg, baseimg)
                diff_scaled = torch.sub(dreamscaled, basescaled)

                self.logger.experiment.add_images("Dream image grid", torch.stack([basescaled, dreamscaled, rescale(diff_scaled, (0, diff_scaled.max() - diff_scaled.min())), rescale(diff_abs)]), global_step = self.img_log_step)
                self.logger.experiment.add_scalar("Mean L2 distance of dreams", torch.linalg.vector_norm(torch.sub(x_dream, x_all_cl), ord=2, dim=(-3, -2, -1)).mean(), global_step = self.img_log_step)
                self.logger.experiment.add_scalar("max difference of dream",  diff_abs.max(), global_step = self.img_log_step)
                self.logger.experiment.add_scalar("min difference of dream", diff_abs.min(), global_step = self.img_log_step)

            x = x_dream.detach().clone()

        xmin, xmax = torch.min(x), torch.max(x)
        assert xmin >= 0 and xmax <= 1, f"x not scaled properly. min and max are {xmin}, {xmax} "

        # Make absolutely sure no grad is tracked.
        if self.model.fc1.weight.grad is not None:
            nonzero_count = torch.count_nonzero(self.model.fc1.weight.grad)
            assert nonzero_count == 0, f"grad is nonzero: count of {nonzero_count} in {self.model.fc1.weight.grad}"

        self.model.train()
        # Normal/regular training cycle

        if not self.with_adversarial:
            # Re-enable gradient tracking for training
            for p in self.model.parameters():
                p.requires_grad_(True)


            y_hat = self.model(x)
            
            loss = self.criterion(y_hat, y)
            acc = self.accuracy_train(y_hat, y)

            self.log("train_loss", loss, on_epoch=True)
            self.log("train_acc", acc, on_epoch=True)
            self.manual_backward(loss=loss)
            opt.step()

        # Adversarial training cycle
        else:
            # Grad tracking is already disabled
            x_adv = self.make_adversarial(x, y)

            # Enable grad tracking:
            for p in self.model.parameters():
                p.requires_grad_(True)

            y_hat_adv = self.model(x_adv)

            loss_adv = self.criterion(y_hat_adv, y)
            acc_adv = self.accuracy_train_adv(y_hat_adv, y)

            # Logging
            self.log("train_adv_loss", loss_adv, on_epoch=True)
            self.log("train_adv_acc", acc_adv, on_epoch=True)
            self.manual_backward(loss = loss_adv)
            opt.step()

        # torch.nn.utils.clip_grad.clip_grad_value_(self.model.parameters(), 1e-7)
        # torch.nn.utils.clip_grad.clip_grad_norm_(self.model.parameters(), 1e0, norm_type=2, error_if_nonfinite=True)

            if (batch_idx % self.log_every_x == 0) and self.attack is not None:
                baseimg = x[0].detach().clone()
                advimg = x_adv[0].detach().clone()
                advscaled = rescale(advimg)
                basescaled = rescale(baseimg)
                diff = torch.sub(advimg, baseimg)
                self.logger.experiment.add_images("Adversarial Training grid", torch.stack([basescaled, advscaled, rescale(diff, (0, diff.max() - diff.min())), rescale(diff)]), global_step=self.img_log_step)
                self.logger.experiment.add_scalar("L2 norm difference of adv attack", torch.linalg.vector_norm(diff, ord=2), global_step = self.img_log_step)

        if (self.with_adversarial or self.dream_param_dict) and (batch_idx % self.log_every_x == 0):
            self.img_log_step += 1
        return loss_adv if self.with_adversarial else loss

    def on_validation_model_eval(self, *args, **kwargs):
        super().on_validation_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True) # Required for adversarial attack generation.

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        x, y = batch
        # disable gradient tracking on model at all times during validation:
        for p in self.model.parameters():
            p.requires_grad_(False)

        xmin, xmax = torch.min(x), torch.max(x)
        assert xmin >= 0 and xmax <= 1, f"x not scaled properly. min and max are {xmin}, {xmax} "

        y_hat = self.model(x)
        
        val_loss = self.criterion(y_hat, y)
        val_acc = self.accuracy_val(y_hat, y)

        self.log("val_loss", val_loss, on_epoch=True)
        self.log("val_acc", val_acc, on_epoch=True)

        if self.attack is not None:
            x_adv = self.make_adversarial(x, y)

            y_hat_adv = self.model(x_adv)

            loss_adv = self.criterion(y_hat_adv, y)
            acc_adv = self.accuracy_val_adv(y_hat_adv, y)
            # Logging
            self.log("val_adv_loss", loss_adv, on_epoch=True)
            self.log("val_adv_acc", acc_adv, on_epoch=True)
            self.log("val_loss_combined", torch.add(loss_adv, val_loss), on_epoch=True)

            if (batch_idx <= 2):
                baseimg = x[0].detach().clone()
                advimg = x_adv[0].detach().clone()
                advscaled = rescale(advimg)
                basescaled = rescale(baseimg)
                diff = torch.sub(advscaled, basescaled)
                self.logger.experiment.add_images("Adversarial validation examples", torch.stack([basescaled, advscaled, rescale(diff, (0, diff.max() - diff.min())), rescale(diff)]), global_step=self.val_img_log_step)
                self.val_img_log_step += 1


    def configure_optimizers(self):
        # optimizer_adv = torch.optim.Adam(self.parameters(), lr=1e-3, betas=[0.5, 0.9]) # Suggested betas by Gennaro
        optimizer = torch.optim.Adam(self.parameters(), lr=self.opt_lrs, weight_decay=self.opt_wd)

        return optimizer

    def make_adversarial(self, x, y):
        x_, y_ = x.detach().clone(), y.clone().detach()

        # Ensure eval mode, revert mode afterwards if model was in training mode
        prev_model_train = self.model.training
        self.model.eval()

        fcriteria = fb.criteria.Misclassification(y_)
        epsilons = self.epsilons_adv
        raw_advs, clipped_advs, success = self.attack(self.fmodel, x_, epsilons=epsilons, criterion=fcriteria)

        self.model.train(mode=prev_model_train)
        return clipped_advs[-1].detach().clone()



    def generate_dream_batch(self, batch):
        dream_imgs = self.dream_generator.generate_dreams(batch=batch)
        return dream_imgs
