from os import path

import pytorch_lightning as pl

import foolbox as fb

import torch

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import warnings


from DataSet.lightning_fashionmnist import F_MNISTDataModule
from DataSet.lightning_cifar import CIFARDataModule
from DataSet.lightning_MNIST import MNISTDataModule
from models.cifar_nn import Cifar_nn
from models.cifar_nn_nobn import Cifar_nn_nobn
from models.fmnist_nn import Fmnist_nn
from models.mnist_nn import Mnist_nn
from lightning_trainer import LitModelTrainer
from models.mnist_nn_nobn import Mnist_nn_nobn


def main(hparams:dict, trainer_kwargs:dict = {}, model_kwargs:dict = {}, dream_kwargs:dict = {}, attack = None):

    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    # num_epochs:int = 50, seed:int = 42, early_stopping:bool = True, early_stop_monitoring:str="val_loss", model_identifier:str = "base", version_nr:int = 0
    # Seed everything
    pl.seed_everything(hparams["seed"], workers=True)
    
    # List of callbacks to use:
    trainer_callbacks = []

    #  Checkpoint saving the model every epoch, if monitor value is given
    if "save_tag_monitor" in hparams:
        save_checkpoint_callback = ModelCheckpoint( 
            save_top_k=1,
            monitor=hparams["save_tag_monitor"],
            filename="model-{epoch:02d}-{val_acc:.3f}",
        )
        trainer_callbacks.append(save_checkpoint_callback)


    if hparams["early_stopping"]:
        early_stop_callback = EarlyStopping(monitor=hparams["early_stop_monitor"], mode="min", patience=10)
        trainer_callbacks.append(early_stop_callback)
    


    if "Cifar" in model_kwargs["nn_arch"]:
        data = CIFARDataModule(num_workers=0, batch_size=hparams["batch_size"])
        if "Cifar-nobn" in model_kwargs["nn_arch"]:
            model = Cifar_nn_nobn
        else:
            model = Cifar_nn

    elif "Mnist" in model_kwargs["nn_arch"]:
        data = MNISTDataModule(num_workers=0, batch_size=hparams["batch_size"])
        if "nobn" in model_kwargs["nn_arch"]:
            model = Mnist_nn_nobn
        else:
            model = Mnist_nn
            

    elif model_kwargs["nn_arch"] == "Fashion_mnist":
        data = F_MNISTDataModule(num_workers=0, batch_size=hparams["batch_size"])
        model = Fmnist_nn


    LITmodel = LitModelTrainer(
        model=model,
        dream_gen_dict=dream_kwargs,
        attack=attack,
        **model_kwargs
    )

    tblogger = TensorBoardLogger(save_dir="trained_models/", name=hparams["model_identifier"], version=hparams["version_nr"])

    trainer = pl.Trainer(logger=tblogger,
        accelerator="gpu",
        # accelerator="cpu",
        max_epochs=hparams["max_num_epoch"],
        amp_backend="apex",
        amp_level='O2',
        callbacks=trainer_callbacks,
        **trainer_kwargs
    )
    
    trainer.validate(model=LITmodel, datamodule=data)
    trainer.fit(model=LITmodel, datamodule=data)


if __name__ == '__main__':
    # Example args / single model run setup:
    _trainer_arg = {
        "deterministic": False,
        "num_sanity_val_steps": 0,
        "log_every_n_steps": 10
    }

# Seeds models: 52 - 57 

    hyper_param = {
        "model_identifier": "CIFAR/PGD/testdreamiterlimit",
        
        "version_nr": 0,
        "seed": 52,
        "early_stopping": True,
        "early_stop_monitor": "val_adv_loss",
        "save_tag_monitor": "val_adv_loss",
        # "early_stop_monitor": "val_loss",
        # "save_tag_monitor": "val_loss",
        "batch_size": 256,
        "max_num_epoch": 50
      }

    model_params = {
        "nn_arch": "Cifar", 
        "with_adversarial": True,
        # "adv_eps": 1.6,
        # "adv_eps": 0.03,
        # "adv_eps": 4.2,
        "adv_eps": 0.03,
        "opt_lrs": 1e-3,
        # "opt_lrs": 1e-3,
        "opt_wd": 0
    }

    # torch.linalg.vector_norm(torch.sub(_initial_other_logits, other_logits), ord=2)
    def penalty_function_l2(new, old):
        loss_value = torch.linalg.vector_norm(torch.sub(new, old), ord=2)
        return loss_value

    # elu function on diff
    def penalty_function_elu(new, old):
        loss_value = torch.functional.F.elu(torch.sub(new, old)).mean(dim=1)
        return loss_value

    dream_params = {
    "iterations": (16,),
    "opt_lr":1e-3,

    # "parametrization": "bound_hyperbolic",
    # "limit_eps": 0.2,

    # "penal_f": penalty_function_l2
    # "penal_f": penalty_function_elu,
    # "penal_factor":1.5,
    }


    # _attack = fb.attacks.deepfool.LinfDeepFoolAttack(steps=25, candidates=3)
    # _attack = fb.attacks.FGSM(random_start=False)

    # Rel stepsize for up and down eps ball = 1/(0.5*steps) -> relative to epsilon
    steps = 5
    _attack = fb.attacks.LinfPGD(steps=steps, rel_stepsize=2/steps)
    # _attack = fb.attacks.LinfPGD(steps=20, rel_stepsize=1/10)

    # _attack = fb.attacks.carlini_wagner.L2CarliniWagnerAttack(binary_search_steps=1, steps=50, stepsize=0.02, initial_const=1, abort_early=True)
    # _attack = fb.attacks.carlini_wagner.L2CarliniWagnerAttack(steps=20, stepsize=0.1, binary_search_steps=1, initial_const=1e0)
    # _attack = None

    main(hparams=hyper_param, trainer_kwargs=_trainer_arg, model_kwargs=model_params, dream_kwargs=dream_params, attack=_attack)


