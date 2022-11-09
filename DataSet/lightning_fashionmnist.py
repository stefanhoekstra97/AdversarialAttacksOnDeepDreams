from typing import Optional
import pytorch_lightning as pl
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms


class F_MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = '../../data/f_mnist', batch_size: int = 256, num_workers: int = 0, shuffle_data: bool = True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_data = shuffle_data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def setup(self, stage: Optional[str] = None):
        self.mnist_test = FashionMNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
        self.mnist_predict = FashionMNIST(root=self.data_dir, train=False, download=True, transform=self.transform)
        self.mnist_full = FashionMNIST(self.data_dir, train=True, download=True, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_full, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_data, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_data, drop_last=True)

    def predict_dataloader(self):
        return DataLoader(self.mnist_predict, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle_data, drop_last=True)

