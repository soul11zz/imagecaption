from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy
import os, sys
import torch

import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)

is_win = sys.platform.startswith("win")
# if we are training with SageMaker

def get_trainer_env():
  
  env = LightningEnvironment()
  
  if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
    
    # Lighning Environment requires this set or else it will assume we are using
    # an internal launcher and bad things will happen
    if "LOCAL_RANK" not in os.environ:
      os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    
    # Environment variables set by mpirun
    LOCAL_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    WORLD_SIZE = lambda: int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    WORLD_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    NODE_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_NODE_RANK', 0))

  
    LOCAL_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
    WORLD_SIZE = lambda: int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
    WORLD_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
    NODE_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_NODE_RANK', 0))
    
    env.world_size = WORLD_SIZE
    env.global_rank = WORLD_RANK
    env.local_rank = LOCAL_RANK
    env.node_rank = NODE_RANK

  assert 'LOCAL_RANK' in os.environ, 'LOCAL_RANK not set'
  return env

def get_initialization_info():
  '''
  Initialize the distributed training environment and return the data relevant to
  Lighning Trainer initialization.
  '''
  
  world_size = num_nodes = num_gpus = 1
  ddp = None

  if not is_win and ("LOCAL_RANK" in os.environ or "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ):
    
    # For DDP with sagemaker
    env = get_trainer_env()
    
    ddp = DDPStrategy(
      cluster_environment=env,
    )

    world_size = env.world_size()
    
    num_gpus = torch.cuda.device_count()
    num_nodes = int(world_size/num_gpus)
    
    logging.info(f"Training with {num_gpus} GPUs/node on {num_nodes} nodes")
    
  return ddp, num_gpus, num_nodes

def get_global_rank():
  env = get_trainer_env()
  return env.global_rank()

def get_local_rank():
  env = get_trainer_env()
  return env.local_rank()