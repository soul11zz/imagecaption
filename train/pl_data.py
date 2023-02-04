import pytorch_lightning as pl
from dataset import ImageCaptioningDataset
import os
from torch.utils.data import DataLoader
from datasets import load_dataset

class ImageCaptionDataModule(pl.LightningDataModule):
  
  def __init__(self, dataset_path, processor, **kwargs):
    super().__init__()
    
    self.dataset_path = dataset_path
    self.processor = processor
    self.prepare_data_per_node = True
    
    self.save_hyperparameters(kwargs)
    self.batch_size = self.hparams.get("batch_size", 1)
    self.auth_token = self.hparams.get("auth_token", None)
  
  def prepare_data(self):
    load_dataset(self.dataset_path, split="train")
  
  def setup(self, stage=None):

    processor = self.processor
    
    if stage != "test":
      dt_train = load_dataset(self.dataset_path, split="train", use_auth_token=self.auth_token)
      self.train_dataset = ImageCaptioningDataset(dt_train, processor)
      
      dt_val = load_dataset(self.dataset_path, split="validation", use_auth_token=self.auth_token)
      self.val_dataset = ImageCaptioningDataset(dt_val, processor)
    else:
      dt_test = load_dataset(self.dataset_path, split="test", use_auth_token=self.auth_token)
      self.test_dataset = ImageCaptioningDataset(dt_test, processor)
      
    self.num_workers = os.cpu_count() if os.name != "nt" else 0
  
  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
  
  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
    