from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.strategies import DDPStrategy
import os, sys

import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)

is_win = sys.platform.startswith("win")
# if we are training with SageMaker

def get_trainer_env():
  env = LightningEnvironment()
  
  LOCAL_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', 0))
  WORLD_SIZE = lambda: int(os.environ.get('OMPI_COMM_WORLD_SIZE', 1))
  WORLD_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_RANK', 0))
  NODE_RANK = lambda: int(os.environ.get('OMPI_COMM_WORLD_NODE_RANK', 0))
  
  env.world_size = WORLD_SIZE
  env.global_rank = WORLD_RANK
  env.local_rank = LOCAL_RANK
  env.node_rank = NODE_RANK
  
  return env

def get_initialization_info():
  '''
  Initialize the distributed training environment and return the data relevant to
  Lighning Trainer initialization.
  '''
  
  world_size = num_nodes = num_gpus = 1
  ddp = None

  if not is_win:
    
    # For DDP with sagemaker
    env = get_trainer_env()
    
    ddp = DDPStrategy(
      cluster_environment=env,
    )

    world_size = env.world_size()
    num_gpus = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", 1))
    num_nodes = int(world_size/num_gpus)
    
    logging.info(f"Training with {num_gpus} GPUs/node on {num_nodes} nodes")
    
  return ddp, num_gpus, num_nodes

def get_global_rank():
  return int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))

def get_local_rank():
  return int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))