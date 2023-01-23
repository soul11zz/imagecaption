from datasets import load_dataset
from dataset import ImageCaptioningDataset

from argparsing import parse_args
from transformers import GitProcessor, GitForCausalLM
from hf_hub_lightning import HuggingFaceHubCallback

from huggingface_hub import HfApi

import torch
from torch.utils.data import DataLoader
from pl_module import ImageCaptioningModule

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

def get_input_model_name(args):
  api = HfApi()
  repo_name = "soul11zz/image-caption-desc-only"
  try:
    api.create_repo(repo_name, private=True, exist_ok=False)
    return "microsoft/git-base"
  except Exception as e:
    return args.model

def training_loop(args):
  
  dt_train = load_dataset(args.train, split="train")
  dt_val = load_dataset(args.val, split="validation")

  input_model_repo = get_input_model_name(args)
  processor = GitProcessor.from_pretrained(input_model_repo)
  
  train_dataset = ImageCaptioningDataset(dt_train, processor)
  val_dataset = ImageCaptioningDataset(dt_val, processor)
    
  train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
  
  model = GitForCausalLM.from_pretrained(input_model_repo)
  callbacks = []
  
  MAX_LR = 1e-2
  pl_train_module = ImageCaptioningModule(processor, model, train_loader, 
                                          val_loader, learning_rate=MAX_LR)
  
  logger = TensorBoardLogger("tb-logs", name="image-captioning")
  
  if args.model:
    callbacks.append(HuggingFaceHubCallback(args.model))
  
  trainer = pl.Trainer(logger=logger, 
                       gpus=1,
                       callbacks=callbacks,
                       max_epochs=args.epochs,
                       check_val_every_n_epoch=1,
                       precision=16,
                       num_sanity_val_steps=1,
                       )

  # find our own learning rate
  lr_finder = trainer.tuner.lr_find(pl_train_module, min_lr=1e-7, max_lr=MAX_LR)
  pl_train_module.lr = lr_finder.suggestion()
  
  trainer.fit(pl_train_module)
   
if __name__ == "__main__":
  args = parse_args()

  training_loop(args)