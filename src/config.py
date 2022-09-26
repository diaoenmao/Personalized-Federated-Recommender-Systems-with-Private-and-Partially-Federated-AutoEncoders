# -*- coding: UTF-8 -*-
import yaml

global cfg
"""
control:
  data_name: ML100K (Name of the dataset)
  data_mode: user (user or item)
  target_mode: ex (explicit(ex) or implicit(im))
  train_mode: fedavg (joint, fedavg)
  federated_mode: de (all or decoder(de))
  model_name: ae 
  data_split_mode: 'iid' (iid or non-iid)
  num_nodes: 100 (1, 100, 300, max)
  compress_transmission: 1 (1: compress, 0: not compress)
  experiment_size: 'l' (l(l): transfer parameters to cpu)

# experiment
num_workers: 0
init_seed: 0
num_experiments: 1
log_interval: 0.25
device: cuda
resume_mode: 0
verbose: False
"""
import sys
import os

cwd = os.getcwd()
if 'cfg' not in globals():
    with open('config.yml', 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)


def process_args(args):
    """
    Update cfg which now is the default yml to args typed in by user.

    Parameters:
        args - Dict. The command typed in by user after vars(parser.parse_args())

    Returns:
        None

    Raises:
        None
    """
    for k in cfg:
        cfg[k] = args[k]

    # if control_name in args is not none, which means the user type in the control_name command.
    # Update the value of cfg['control']
    if 'control_name' in args and args['control_name'] is not None:
        control_name_list = args['control_name'].split('_')
        control_keys_list = list(cfg['control'].keys())
        cfg['control'] = {control_keys_list[i]: control_name_list[i] for i in range(len(control_name_list))}
    
    # Update the value of cfg['control_name']
    if cfg['control'] is not None:
        cfg['control_name'] = '_'.join([str(cfg['control'][k]) for k in cfg['control']])
    return
