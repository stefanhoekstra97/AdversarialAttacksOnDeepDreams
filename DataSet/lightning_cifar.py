from typing import Optional
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '../../../data/cifar', batch_size: int = 256, num_workers: int = 0, shuffle_data: bool = True, pin_memory: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_data = shuffle_data
        self.pin = pin_memory
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def setup(self, stage: Optional[str] = None):
        # self.cifar_test = CIFAR100(self.data_dir, train=False, download=True, transform=self.transform)
        self.cifar_test = CIFAR10(self.data_dir, train=False, download=True, transform=self.transform)
        # self.cifar_test =  torch.utils.data.Subset(self.cifar_test, range(5000, 9999))
        # self.cifar_predict = CIFAR100(self.data_dir, train=False, download=True, transform=self.transform)
        self.cifar_predict = CIFAR10(self.data_dir, train=False, download=True, transform=self.transform)

        # self.cifar_full = CIFAR100(self.data_dir, train=True, download=True, transform=self.transform)
        self.cifar_full = CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.cifar_full, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_data, drop_last=True, pin_memory=self.pin)

    def val_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True, pin_memory=self.pin)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_data, drop_last=True, pin_memory=self.pin)

    def predict_dataloader(self):
        return DataLoader(self.cifar_predict, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_data, drop_last=True, pin_memory=self.pin)
