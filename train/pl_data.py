import pytorch_lightning as pl
from dataset import ImageCaptioningDataset
import os
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from distributed import get_global_rank, get_world_size

# to avoid running out of file descriptors for dataloader workers
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class ImageCaptionDataModule(pl.LightningDataModule):

    def __init__(self, dataset_path, processor, auth_token=None, **kwargs):
        super().__init__()

        self.dataset_path = dataset_path
        if isinstance(dataset_path, str):
            self.dataset_path = [dataset_path]

        self.processor = processor
        self.prepare_data_per_node = True

        self.save_hyperparameters(kwargs)
        self.batch_size = self.hparams.get("batch_size", 1)

        # pass that to fine-tune the number of workers
        self.num_gpus = self.hparams.get("num_gpus", 1)

        self.auth_token = auth_token
        self.train_loader = self.val_loader = self.test_loader = None
        self.num_workers = min(12, os.cpu_count() //
                               self.num_gpus) if os.name != "nt" else 0

    def prepare_data(self):

        if self.train_loader is not None:
            return
        for dataset_path in self.dataset_path:
            load_dataset(dataset_path, split="train", use_auth_token=self.auth_token, num_proc=self.num_workers + 1)

    def load_datasets(self, split):
        dataset_list = []
        for dataset_path in self.dataset_path:
            dt = load_dataset(dataset_path, split=split, use_auth_token=self.auth_token, num_proc=self.num_workers + 1)
            dataset_list.append(ImageCaptioningDataset(dt, self.processor))
        return ConcatDataset(dataset_list)
    
    def setup(self, stage=None):

        if stage != "test":
            # we have already downloaded the train dataloader
            if self.train_loader:
                return

            self.train_dataset = self.load_datasets("train")
            self.val_dataset = self.load_datasets("validation")

            self.tran_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        else:
            self.test_dataset = self.load_datasets("test")
            self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)

    def train_dataloader(self):
        return self.tran_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
