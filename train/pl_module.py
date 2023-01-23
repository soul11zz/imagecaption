from pathlib import Path
import re
from nltk import edit_distance
import numpy as np
import math
import torch

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only


class ImageCaptioningModule(pl.LightningModule):
    def __init__(self, processor, model, train_dataloader, val_dataloader, learning_rate=None):
        super().__init__()
        self.model = model
        self.processor = processor
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
      input_ids = batch["input_ids"]
      pixel_values = batch["pixel_values"]

      outputs = self.model(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)
    
      loss = outputs.loss
      self.log_dict({"train_loss": loss}, sync_dist=True)
      return loss

    def validation_step(self, batch, batch_idx, dataset_idx=0):

      input_ids = batch["input_ids"]
      pixel_values = batch["pixel_values"]

      outputs = self.model(input_ids=input_ids,
                      pixel_values=pixel_values,
                      labels=input_ids)
    
      loss = outputs.loss
      try:
        perplexity = torch.exp(loss)
      except OverflowError:
        perplexity = float("inf")
        
      self.log_dict({"val_loss": loss, "perplexity": perplexity}, sync_dist=True)
      return loss

    def configure_optimizers(self):
        # TODO add scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    
        return optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
      