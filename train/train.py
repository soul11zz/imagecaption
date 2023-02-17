import sys
import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)
import os.path as osp
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
import distributed as dist
  
def crate_data_module(dataset_name, processor, num_gpus, batch_size, auth_token):
  
  return ImageCaptionDataModule(dataset_name, processor, num_gpus = num_gpus, batch_size=batch_size, auth_token=auth_token)
  
  
def training_loop(args):
  
  ddp, num_gpus, num_nodes = dist.get_initialization_info()
  
  pl.seed_everything(42, workers=True)
  hf_token = os.environ["HF_AUTH_TOKEN"]
  
  input_model_repo = args.model
  processor = GitProcessor.from_pretrained(input_model_repo, use_auth_token=hf_token)
  
  data_module = crate_data_module(args.dataset, processor, num_gpus, args.batch_size, hf_token)
  
  # We need to make sure that all the data is downloaded before we start training
  # otherwise we may run into NCCL timeout issues during distributed training
  if num_gpus > 1:
    if dist.get_local_rank() == 0:
      data_module.prepare_data()
      
    data_module.setup()
    env = ddp.cluster_environment
    torch.distributed.init_process_group(backend="nccl", rank=env.global_rank(), world_size=env.world_size())
    torch.distributed.barrier()

  model = GitForCausalLM.from_pretrained(input_model_repo, use_auth_token=hf_token)
  pl_train_module = ImageCaptioningModule(processor, model, learning_rate=args.lr)

  # Tune learning rate only and quit
  if args.tune_lr:
    logging.info("Tuning learning rate...")
    tuner = pl.Trainer(auto_lr_find=True, devices=1, accelerator="cuda", num_sanity_val_steps=0)
    tuner.tune(pl_train_module, datamodule=data_module)
    return
  
  callbacks = []
  
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
                       devices=num_gpus,
                       num_nodes=num_nodes,
                       strategy=ddp,
                       accelerator="cuda",
                       callbacks=callbacks,
                       max_epochs=args.epochs,
                       check_val_every_n_epoch=1,
                       # val_check_interval=50,
                       precision=16,
                       num_sanity_val_steps=0,
                       )

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
    trainer.test(pl_model_best, datamodule=data_module)
  
  return checkpoint.best_model_path
 
def save_best_model(pl_best_model, model_repo, save_dir = "tb_logs/image-captioning/best_model"):
    
    if not osp.exist(save_dir):
      os.makedirs(save_dir)
      
    pl_best_model.model.save_pretrained(save_dir, push_to_hub=True, repo_id=model_repo)
    pl_best_model.processor.save_pretrained(save_dir, push_to_hub=True, repo_id=model_repo)
    return pl_best_model

    
if __name__ == "__main__":
  args = parse_args()

  training_loop(args)