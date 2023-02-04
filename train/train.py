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
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from pl_data import ImageCaptionDataModule
  
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

def crate_data_module(dataset_name, processor, batch_size, auth_token):
  
  return ImageCaptionDataModule(dataset_name, processor, batch_size=batch_size, auth_token=auth_token)
  
  
def training_loop(args):
  
  pl.seed_everything(42, workers=True)
  
  input_model_repo = get_input_model_name(args.model)
  processor = GitProcessor.from_pretrained(input_model_repo)
  
  data_module = crate_data_module(args.dataset, processor, args.batch_size, os.getenv("HF_AUTH_TOKEN", None))
  
  model = GitForCausalLM.from_pretrained(input_model_repo)
  callbacks = []
  
  pl_train_module = ImageCaptioningModule(processor, model, learning_rate=1e-2)
  
  ### Trainer
  logger = TensorBoardLogger("tb_logs", name="image-captioning")
  
  checkpoint = ModelCheckpoint(dirpath=args.model_dir, 
                               save_top_k=2, monitor="val_loss", 
                               mode="min", 
                               filename="imcap-{epoch:02d}-{val_loss:.2f}")
  
  early_stopping = EarlyStopping(monitor="val_loss", patience=3, mode="min")
  
  lr_monitor = LearningRateMonitor(logging_interval="step")
  callbacks += [checkpoint, early_stopping, lr_monitor]
  
  trainer = pl.Trainer( 
                        logger=logger, 
                       devices=1,
                       accelerator="cuda",
                       callbacks=callbacks,
                       max_epochs=args.epochs,
                       check_val_every_n_epoch=1,
                       # val_check_interval=50,
                       precision=16,
                       num_sanity_val_steps=0,
                       )

  # find our own learning rate
  logging.info("Tuning learning rate...")
  
  if args.tune_lr:
    logging.info("Tuning learning rate...")
    tuner = pl.Trainer(auto_lr_find=True, devices=1, accelerator="cuda", num_sanity_val_steps=0)
    tuner.tune(pl_train_module, datamodule=data_module)
  
  # and fit
  logging.info("Training...")
  
  trainer.fit(pl_train_module, datamodule=data_module)
  
  pl_model_best = None
  if args.save_best or args.test_best:
    pl_model_best = ImageCaptioningModule.load_from_checkpoint(checkpoint.best_model_path, processor=processor, model=model,)
  
  # save best model
  if args.save_best:
    logging.info("Saving best model...")
    if not args.best_model:
      logging.warning("No best model name provided, skipping upload of the best model to HuggingFace Hub.")
    else:
      logging.info(f"Uploading best model to HuggingFace Hub as {args.best_model}")
      save_best_model(pl_model_best, args.best_model)

  if args.test_best:
    logging.info("Testing best model...")
    trainer.test(pl_model_best)
  
  return checkpoint.best_model_path
 
def save_best_model(model_repo, pl_best_model):
    
    pl_best_model.model.save_pretrained("tb_logs/image-captioning/best_model", push_to_hub=True, repo_id=model_repo)
    pl_best_model.processor.save_pretrained("tb_logs/image-captioning/best_model", push_to_hub=True, repo_id=model_repo)
    return pl_best_model

    
if __name__ == "__main__":
  args = parse_args()

  training_loop(args)