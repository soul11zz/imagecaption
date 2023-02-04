from datasets import load_dataset
from dataset import ImageCaptioningDataset
import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)

from argparsing import parse_args
from transformers import GitProcessor, GitForCausalLM

from huggingface_hub import HfApi
from pl_data import ImageCaptionDataModule

import torch
from torch.utils.data import DataLoader
from pl_module import ImageCaptioningModule
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def score_model(model_name, dataset_name, metric):
    model = GitForCausalLM.from_pretrained(model_name)
    processor = GitProcessor.from_pretrained(model_name)
    pl_data_module = ImageCaptionDataModule(dataset_name, processor, batch_size=1, auth_token=os.getenv("HF_AUTH_TOKEN", None))    
    pl_train_module = ImageCaptioningModule(processor, model, metric=metric)
    
    tester = pl.Trainer(accelerator="cuda", devices=1, num_sanity_val_steps=0)
    
    tester.test(pl_train_module, pl_data_module)

if __name__ == "__main__":
  args = parse_args()

  score_model(args.model, args.dataset, args.metric)