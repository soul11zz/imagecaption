import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)

from argparsing import parse_args
from transformers import GitProcessor, GitForCausalLM

from pl_data import ImageCaptionDataModule

from pl_module import ImageCaptioningModule
import os

import pytorch_lightning as pl

def score_model(model_name, dataset_name, metric):
    hf_token = os.environ["HF_AUTH_TOKEN"]
    model = GitForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
    processor = GitProcessor.from_pretrained(model_name, use_auth_token=hf_token)
    pl_data_module = ImageCaptionDataModule(dataset_name, processor, batch_size=1, auth_token=hf_token)    
    pl_train_module = ImageCaptioningModule(processor, model, metric=metric)
    
    tester = pl.Trainer(accelerator="cuda", devices=1, num_sanity_val_steps=0)
    
    tester.test(pl_train_module, pl_data_module)

if __name__ == "__main__":
  args = parse_args()

  score_model(args.model, args.dataset, args.metric)