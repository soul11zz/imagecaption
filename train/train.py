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
  
def get_input_model_name(model_name_with_repo):
  api = HfApi()
  repo_name = model_name_with_repo
  try:
    api.create_repo(repo_name, private=True, exist_ok=False)
    return "microsoft/git-base"
  except:
    try:
      GitProcessor.from_pretrained(repo_name)
      return model_name_with_repo
    except:
      return "microsoft/git-base"

def training_loop(args):
  
  pl.seed_everything(42, workers=True)
  
  dt_train = load_dataset(args.train, split="train")
  dt_val = load_dataset(args.val, split="validation")
  dt_test = load_dataset(args.test, split="test")
  
  input_model_repo = get_input_model_name(args.model)
  processor = GitProcessor.from_pretrained(input_model_repo)
  
  train_dataset = ImageCaptioningDataset(dt_train, processor)
  val_dataset = ImageCaptioningDataset(dt_val, processor)
  test_dataset = ImageCaptioningDataset(dt_test, processor)
  
  num_workers = os.cpu_count() if os.name != "nt" else 0
  train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=num_workers)
  val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=num_workers)
  test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)
  
  model = GitForCausalLM.from_pretrained(input_model_repo)
  callbacks = []
  
  pl_train_module = ImageCaptioningModule(processor, model, train_loader, val_loader, test_loader, learning_rate=1e-2)
  
  ### Trainer
  logger = TensorBoardLogger("tb_logs", name="image-captioning")
  
  checkpoint = ModelCheckpoint(dirpath=args.model_dir, 
                               save_top_k=2, monitor="val_loss", 
                               mode="min", 
                               filename="imcap-{epoch:02d}-{val_loss:.2f}")
  
  callbacks += [checkpoint]
  
  trainer = pl.Trainer( 
                        logger=logger, 
                       devices=1,
                       accelerator="cuda",
                       callbacks=callbacks,
                       max_epochs=args.epochs,
                       check_val_every_n_epoch=1,
                       # val_check_interval=50,
                       precision=16,
                       num_sanity_val_steps=2,
                       )

  # find our own learning rate
  logging.info("Tuning learning rate...")
  
  if args.tune_lr:
    logging.info("Tuning learning rate...")
    tuner = pl.Trainer(auto_lr_find=True, devices=1, accelerator="cuda", num_sanity_val_steps=0)
    tuner.tune(pl_train_module)
  
  # and fit
  logging.info("Training...")
  
  trainer.fit(pl_train_module)
  
  # save best model
  if args.save_best:
    logging.info("Saving best model...")
    save_best_model(args.model, processor, train_loader, val_loader, model, checkpoint)

def save_best_model(model_repo, processor, train_loader, val_loader, test_dataloader, model, checkpoint):
    pl_model_best = ImageCaptioningModule.load_from_checkpoint(checkpoint.best_model_path, processor=processor, model=model, train_dataloader=train_loader, val_dataloader=val_loader, test_dataloader=test_dataloader)
    pl_model_best.model.save_pretrained("tb_logs/image-captioning/best_model", push_to_hub=True, repo_id=model_repo)
    pl_model_best.processor.save_pretrained("tb_logs/image-captioning/best_model", push_to_hub=True, repo_id=model_repo)
    return pl_model_best

def test_best_model(trainer, pl_model_best):
  
    trainer.test(pl_model_best)
    
if __name__ == "__main__":
  args = parse_args()

  training_loop(args)