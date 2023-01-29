from datasets import load_dataset
from dataset import ImageCaptioningDataset
import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)

from argparsing import parse_args
from transformers import GitProcessor, GitForCausalLM

from huggingface_hub import HfApi

import torch
from torch.utils.data import DataLoader
from pl_module import ImageCaptioningModule
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def score_model(model_name, dataset_name):
    model = GitForCausalLM.from_pretrained(model_name)
    processor = GitProcessor.from_pretrained(model_name)
    
    dt_test = load_dataset(dataset_name, split="test")
    test_dataset = ImageCaptioningDataset(dt_test, processor)
    
    num_workers = os.cpu_count() if os.name != "nt" else 0
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
    pl_train_module = ImageCaptioningModule(processor, model, train_dataloader=None, val_dataloader=None, test_dataloader=test_loader)
    
    tester = pl.Trainer(accelerator="cuda", devices=1, num_sanity_val_steps=0)
    
    tester.test(pl_train_module)
    
    pass

if __name__ == "__main__":
  args = parse_args()

  score_model(args.model, args.dataset)