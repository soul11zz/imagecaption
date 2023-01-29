from argparse import ArgumentParser
import os

def parse_args():
  parser = ArgumentParser()
  
  # Datasets
  default_train = os.getenv("SM_CHANNEL_TRAIN", None)
  default_val = os.getenv("SM_CHANNEL_VAL", None)
  
  parser.add_argument("--dataset", "-d", type=str, default=default_train,
                      help="Single path to directory of annotated JSON training data")
  
  # Model Hyper-parameters
  default_batches = int(os.getenv('SM_HP_BATCH', 1))
  default_epochs = int(os.getenv('SM_HP_EPOCHS', 1))
  default_check_interval = float(os.getenv('SM_HP_VAL_CHECK_INTERVAL', 0.0))
  if default_check_interval == 0.0:
    default_check_interval = None
    
  default_check_epoch = int(os.getenv('SM_HP_CHECK_VAL_EVERY_N_EPOCH', 0))
  
  if default_check_epoch == 0:
    default_check_epoch = None
    
  default_lr = float(os.getenv('SM_HP_LR', 3e-5))
  default_warmup = int(os.getenv("SM_HP_WARMUP_STEPS", 100))
  
  default_verbose = os.getenv("SM_HP_VERBOSE", False)
  default_verbose = True if default_verbose and default_verbose.lower() == "true" else False
  
  parser.add_argument("--batch", "-b", type=int, required=False, default=default_batches, help="Batch size")
  parser.add_argument("--epochs", "-e", type=int, required=False, default=default_epochs, help="Number of training epochs")
  parser.add_argument("--val_check_interval", type=float, required=False, default=default_check_interval, help="After how many steps to validate")
  parser.add_argument("--check_val_every_n_epoch", type=int, required=False, default=default_check_epoch, help="After how many epochs to validate")
  parser.add_argument("--lr", type=float, required=False, default=default_lr, help="Initial learning rate")
  parser.add_argument("--warmup_steps", type=int, required=False, default=default_warmup, help="Steps before training begins")
  parser.add_argument("--verbose", action="store_true", required=False, default=default_verbose, help="Validation verbosity (print pred/answer)")

  # Auto tuning
  parser.add_argument("--tune-lr", action="store_true", required=False, default=False, help="Tune learning rate with PL")
  
  # Saving best model to hub
  parser.add_argument("--save-best", action="store_true", required=False, default=False, help="Save best model to hub")
  parser.add_argument("--test-best", action="store_true", required=False, default=False, help="Test best model")
  # Results dir
  parser.add_argument("--model_dir", help="Result dir", default=os.getenv("SM_MODEL_DIR", "./tmp"))
  
  # Checkpoint
  parser.add_argument("--ckpt", help="Checkpoint for inference", default=None)
  
  # Model output
  parser.add_argument("--model", "-m", help="Output model name", default=None)
  
  # Metrics (bleu or meteor)
  parser.add_argument("--metric", help="Metric to use for best model (bleu or meteor)", default="bleu")
  
  return parser.parse_args()