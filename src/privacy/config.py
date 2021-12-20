# -*- coding: UTF-8 -*-
import yaml


global cfg

"""
Put default command (in config.yml) into cfg if 'cfg' not in globals()
yml:
    # control
    control:
        data_name: ML100K
        data_mode: user
        target_mode: implicit
        model_name: ae
        info: 1
        data_split_mode: random-2
        ar: constant-0.1
        aw: constant
        match_rate: 1
    # experiment
    train_mode: private
    private_decoder_user: 10
    num_workers: 0
    init_seed: 0
    num_experiments: 1
    log_interval: 0.25
    device: cuda
    world_size: 1
    resume_mode: 0
    verbose: False

Parameters:
    None

Returns:
    None

Raises:
    None
"""

# globals() 函数会以字典类型返回当前位置的全部全局变量。
if 'cfg' not in globals():
    
    # with用于创建一个临时的运行环境，运行环境中的代码执行完后自动安全退出环境。
    with open('config.yml', 'r') as f:
        # cfg: 字典结构，内有control（sub_dict)
        # {'control': {'data_name': 'ML100K', 'data_mode': 'user', 'target_mode': 'implicit', 'model_name': 'ae', 'info': 1, 'data_split_mode': 'random-2', 'ar': 'constant-0.1', 'aw': 'constant', 'match_rate': 1}, 
        # 'num_workers': 0, 'init_seed': 0, 'num_experiments': 1, 'log_interval': 0.25, 'device': 'cuda', 'world_size': 1, 'resume_mode': 0, 'verbose': False}
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
