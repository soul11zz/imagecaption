import os, sys, glob
import os.path as osp
import re
import copy

import logging
logging.basicConfig(format='%(asctime)s  %(levelname)-10s %(message)s', datefmt="%Y-%m-%d-%H-%M-%S", level=logging.INFO)

from argparsing import parse_args

def prep_config(config_sh, ips):
  
  new_config = copy.deepcopy(config_sh)
  for i, line in enumerate(new_config):
    if "HEAD_IP" in line:
      [key, _] = line.split("=")
      new_config[i] = f"{key}={ips[0]}"
    elif "WORKER_IP" in line:
      [key, _] = line.split("=")
      new_config[i] = f"{key}={' '.join(ips[1:])}"
      
  return new_config
      
def modify_run_sh(run_sh, ip_list, gpu_list):
  
  new_run = copy.deepcopy(run_sh)
  master_set = False
  workers_set = False
  n_proc = sum(gpu_list)
  
  for i, line in enumerate(new_run):
    
    if "mpirun" in line:
      tokens = line.split(" ")
      tokens[2] = str(n_proc)
      new_run[i] = " ".join(tokens)
      
    if not master_set and "MASTER_HOST" in line:
      [key, _] = line.split("=")
      new_run[i] = f"{key}={ip_list[0]}"
      master_set = True
          
    elif not workers_set and "WORKERS" in line:
      [key, _] = line.split("=")
      
      workers = ", ".join([f"{ip}:{gpu}" for ip, gpu in zip(ip_list, gpu_list)])
      new_run[i] = f"{key}={workers}"
      workers_set = True
            
    elif "HF_AUTH_TOKEN" in line:
      [key, _] = line.split("=")
      new_run[i] = f"{key}={os.environ['HF_AUTH_TOKEN']}"
      
  return new_run
      
def modify_setup(run_sh_path, config_sh_path, ip_list, gpu_list):
  
  hf_token = os.environ.get("HF_AUTH_TOKEN", None)
  assert hf_token is not None, "HF_AUTH_TOKEN environment variable must be set"
  
  gpus = list(map(int, gpu_list.split(",")))
  ips = ip_list.split(",")
  assert len(ips) == len(gpus), "Number of IPs and number of GPUs must match"
  
  with open(config_sh_path, "r") as f:
    config_sh = f.readlines()
    
  with open(run_sh_path, "r") as f:
    run_sh = f.readlines()
  
  config_lines = prep_config(config_sh)
  run_lines = modify_run_sh(run_sh, ips, gpus)

  logging.info("New config file")
  logging.info(config_lines)
  
  logging.info("New run file")
  logging.info(run_lines)
  
  with open(config_sh_path, "w") as f:
    f.writelines(list(map(lambda x: x + "\n", config_lines)))

  with open(run_sh_path, "w") as f:
    f.writelines(list(map(lambda x: x + "\n", run_lines)))
  

if __name__ == "__main__":
  
  args = parse_args()
  
  modify_setup(args.run_sh_path, args.config_sh_path, args.ip_list, args.gpu_list) 
  
  