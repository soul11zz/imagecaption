import pytorch_lightning as pl
from dataset import ImageCaptioningDataset
import os
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset
from distributed import get_global_rank, get_world_size
class ImageCaptionDataModule(pl.LightningDataModule):
  
  def __init__(self, dataset_path, processor, auth_token=None, **kwargs):
    super().__init__()
    
    self.dataset_path = dataset_path
    self.processor = processor
    self.prepare_data_per_node = True
    
    self.save_hyperparameters(kwargs)
    self.batch_size = self.hparams.get("batch_size", 1)
    
    # pass that to fine-tune the number of workers
    self.num_gpus = self.hparams.get("num_gpus", 1)
    
    self.auth_token = auth_token
    self.train_loader = self.val_loader = self.test_loader = None
    self.num_workers = os.cpu_count() // self.num_gpus if os.name != "nt" else 0
    
  def prepare_data(self):
    load_dataset(self.dataset_path, split="train", use_auth_token=self.auth_token)
  
  def setup(self, stage=None):

    processor = self.processor
    
    if stage != "test":
      # we have already downloaded the train dataloader
      if self.train_loader:
        return
      
      dt_train = load_dataset(self.dataset_path, split="train", use_auth_token=self.auth_token)
      self.train_dataset = ImageCaptioningDataset(dt_train, processor)
      
      dt_val = load_dataset(self.dataset_path, split="validation", use_auth_token=self.auth_token)
      self.val_dataset = ImageCaptioningDataset(dt_val, processor)
      
      # Distributed sampling
      if self.num_gpus > 0:
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=get_world_size(), rank=get_global_rank(), shuffle=True)
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=get_world_size(), rank=get_global_rank(), shuffle=False)
        
      self.tran_loader =  DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, shuffle=False, num_workers=self.num_workers)
      self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler = val_sampler, shuffle=False, num_workers=self.num_workers)
    else:
      dt_test = load_dataset(self.dataset_path, split="test", use_auth_token=self.auth_token)
      self.test_dataset = ImageCaptioningDataset(dt_test, processor)
      self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=self.num_workers)
      
  
  def train_dataloader(self):
    return self.tran_loader
  
  def val_dataloader(self):
    return self.val_loader
  
  def test_dataloader(self):
    return self.test_loader  