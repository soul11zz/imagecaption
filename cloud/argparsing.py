from argparse import ArgumentParser
import os

def parse_args():
  parser = ArgumentParser()

  parser.add_argument("--run-sh-path", "-i", required=True, help="Path to run.sh")
  parser.add_argument("--config-path", "-c", required=True, help="Path to config.sh")
  parser.add_argument("--ip-list",  required=True, help="List of IPs to use, comma separated (first in the list is the master)")
  parser.add_argument("--gpu-list",  required=True, help="List of GPUs/IP to use, comma separated")
  
  return parser.parse_args()