from ast import Pass
import collections.abc as container_abcs
import collections
import errno
import numpy as np
import os
import pickle
import models
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def concatenate_path(path_item_list):
    res = os.path.join(*path_item_list)
    makedir_exist_ok(res)
    
    return res

def change_to_absolute_path(path):
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), path)
    
def check_exists(path):
    path = change_to_absolute_path(path)
    # print('zheli', path, os.path.exists(path))
    # print('sadfas', os.path.exists(os.path.join('.', 'data', 'ML1M', 'raw')), os.path.exists(os.path.join('./data/ML1M', 'processed')))
    # print('wenjianjia', os.path.abspath(os.path.dirname(__file__)))
    # print(os.listdir(path))
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return

def processed_folder(epoch, isSingle_model):
    root = './federated_privacy/'
    model_name = cfg['model_name']
    if isSingle_model:
        root = os.path.join(os.path.expanduser(root), cfg['model_name'], str(cfg[model_name]['fraction']), str(cfg[model_name]['local_epoch']), str(epoch))
    else:
        root = os.path.join(os.path.expanduser(root), cfg['model_name'], str(cfg[model_name]['fraction']), str(cfg[model_name]['local_epoch']), 'combine', str(epoch))

    return root

def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)

    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        # serializing object
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        # deserializing object
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif isinstance(input, int):
        output = input
    elif isinstance(input, float):
        output = input
    elif input is None:
        output = None
    else:
        a = input
        raise ValueError('Not valid input type')
    return output

def process_fine_tune():
    cfg['fine_tune'] = False
    if 'fine_tune' in cfg:
        cfg['fine_tune'] = int(cfg['fine_tune']) == 1
    if 'fine_tune_lr' in cfg:
        cfg['fine_tune_lr'] = float(cfg['fine_tune_lr'])
    if 'fine_tune_batch_size' in cfg:
        cfg['fine_tune_batch_size'] = int(cfg['fine_tune_batch_size'])
    if 'fine_tune_epoch' in cfg:
        cfg['fine_tune_epoch'] = int(cfg['fine_tune_epoch'])
    if 'fix_layers' in cfg:
        cfg['fix_layers'] = cfg['fix_layers']
    if 'fine_tune_scheduler' in cfg:
        cfg['fine_tune_scheduler'] = cfg['fine_tune_scheduler']

    model_name = cfg['model_name']
    if cfg['fine_tune'] == True:
        # cfg[model_name]['momentum'] = 0.9
        # cfg[model_name]['nesterov'] = True
        cfg[model_name]['optimizer_name'] = 'SGD'
        cfg[model_name]['momentum'] = 0
        cfg[model_name]['nesterov'] = False
        cfg[model_name]['betas'] = (0.9, 0.999)
        cfg[model_name]['weight_decay'] = 5e-4
        cfg[model_name]['batch_size'] = {'test': cfg['fine_tune_batch_size']}
        cfg[model_name]['lr'] = cfg['fine_tune_lr']
        cfg[model_name]['num_epochs'] = cfg['fine_tune_epoch']
        cfg[model_name]['scheduler_name'] = cfg['fine_tune_scheduler']
        cfg[model_name]['milestones'] = [10, 20, 30]
        cfg[model_name]['factor'] = 0.1

def process_dataset(dataset):
    
    """
    Add some key:value about dataset(info about the number of datapoints) to cfg

    Parameters:
        dataset - Dict. dataset['train'] is the train dataset instance. dataset['test']
            is the test dataset instance

    Returns:
        None

    Raises:
        None
    """
    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    cfg['num_users'], cfg['num_items'] = dataset['train'].num_users, dataset['train'].num_items
    if cfg['num_nodes'] == 9999999:
        cfg['num_nodes'] = cfg['num_users']['data']
    if cfg['info'] == 1:
        cfg['info_size'] = {}
        # hasattr() determines if the object has corresponding attribute
        if hasattr(dataset['train'], 'user_profile'):
            cfg['info_size']['user_profile'] = dataset['train'].user_profile['data'].shape[1]
        # if hasattr(dataset['train'], 'item_attr'):
        #     cfg['info_size']['item_attr'] = dataset['train'].item_attr['data'].shape[1]
    else:
        cfg['info_size'] = None
    return


def process_control():
    """
    Since cfg is global variable, dont need to pass parameter
    1. Disassemble cfg['control']
    2. Add the model parameters set 

    Parameters:
        None

    Returns:
        None

    Raises:
        None
    """
    cfg['data_name'] = cfg['control']['data_name']
    cfg['data_mode'] = cfg['control']['data_mode']
    # convert some abbreviations to full name
    if cfg['control']['target_mode'] == 'ex':
        cfg['target_mode'] = 'explicit'
    elif cfg['control']['target_mode'] == 'im':
        cfg['target_mode'] = 'implicit'
    cfg['train_mode'] = cfg['control']['train_mode']

    if cfg['control']['federated_mode'] == 'de':
        cfg['federated_mode'] = 'decoder'
    elif cfg['control']['federated_mode'] == 'all':
        cfg['federated_mode'] = 'all'
    elif cfg['control']['federated_mode'] == 'NA':
        cfg['federated_mode'] = 'NA'

    cfg['model_name'] = cfg['control']['model_name']
    cfg['info'] = float(cfg['control']['info']) if 'info' in cfg['control'] else 0
    cfg['data_split_mode'] = cfg['control']['data_split_mode']

    if cfg['control']['update_best_model'] == 'g':
        cfg['update_best_model'] = 'global'
    elif cfg['control']['update_best_model'] == 'l':
        cfg['update_best_model'] = 'local'

    if cfg['control']['num_nodes'] != 'max':
        cfg['num_nodes'] = int(cfg['control']['num_nodes'])
    else:
        cfg['num_nodes'] = 9999999

    if 'compress_transmission' in cfg['control']:
        cfg['compress_transmission'] = int(cfg['control']['compress_transmission']) == 1
    if 'experiment_size' in cfg['control'] and cfg['control']['experiment_size'] == 'l':
        cfg['experiment_size'] = 'large'

    process_fine_tune()

    # Add size of layer of encoder and decoder
    cfg['ae'] = {'encoder_hidden_size': [256, 128], 'decoder_hidden_size': [128, 256]}
    # Add batch_size
    batch_size = {'user': {
                    'ML100K': 100, 
                    'ML1M': 500, 
                    'ML10M': 5000, 
                    'ML20M': 5000, 
                    'Douban': 100,
                    'taobaoclicksmall': 100,
                    'taobaoclickmedium': 500,
                    'taobaoclicklarge': 5000},
                'item': {
                    'ML100K': 100, 
                    'ML1M': 500, 
                    'ML10M': 1000, 
                    'ML20M': 1000, 
                    'Douban': 100,
                    'taobaoclicksmall': 100,
                    'taobaoclickmedium': 500,
                    'taobaoclicklarge': 5000}}

    # add parameter to model
    # Example: cfg['model_name']: ae         
    global_epoch = 800
    local_epoch = 5
    model_name = cfg['model_name']
    cfg['client'] = collections.defaultdict(dict)
    cfg['server'] = collections.defaultdict(dict)

    # cfg['client'][model_name]['shuffle'] = {'train': True, 'test': True}
    cfg['client'][model_name]['shuffle'] = {'train': False, 'test': False}
    if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
        if cfg['train_mode'] == 'fedavg':
            if cfg['num_nodes'] == 1:
                # simulate joint situation
                cfg['client'][model_name]['fraction'] = 1
                cfg['client'][model_name]['local_epoch'] = 1
                cfg['client'][model_name]['optimizer_name'] = 'Adam'
                cfg['client'][model_name]['lr'] = 1e-3
                cfg['client'][model_name]['scheduler_name'] = 'None'
            else:
                batch_size = {'user': {
                                'ML100K': 5, 
                                'ML1M': 10, 
                                'ML10M': 10, 
                                'ML20M': 10, 
                                'Douban': 10,
                                'taobaoclicksmall': 10,
                                'taobaoclickmedium': 10,
                                'taobaoclicklarge': 10,},
                            'item': {
                                'ML100K': 5, 
                                'ML1M': 10, 
                                'ML10M': 10, 
                                'ML20M': 10, 
                                'Douban': 10,
                                'taobaoclicksmall': 10,
                                'taobaoclickmedium': 10,
                                'taobaoclicklarge': 10}}
                cfg['client'][model_name]['fraction'] = 0.1
                cfg['client'][model_name]['local_epoch'] = local_epoch
                cfg['client'][model_name]['optimizer_name'] = 'SGD'
                cfg['client'][model_name]['lr'] = 0.1
                cfg['client'][model_name]['scheduler_name'] = 'CosineAnnealingLR'
        elif cfg['train_mode'] == 'joint':
            cfg['client'][model_name]['fraction'] = 1
            cfg['client'][model_name]['optimizer_name'] = 'Adam'
            cfg['client'][model_name]['lr'] = 1e-3
            cfg['client'][model_name]['scheduler_name'] = 'None'
        # elif cfg['train_mode'] == 'fedsgd':
        #     cfg['client'][model_name]['fraction'] = 0.1
        #     cfg['client'][model_name]['local_epoch'] = local_epoch
        #     cfg['client'][model_name]['optimizer_name'] = 'SGD'
        #     cfg['client'][model_name]['lr'] = 0.1
        #     cfg['client'][model_name]['scheduler_name'] = 'CosineAnnealingLR'
    elif cfg['data_name'] == 'Douban':
        # print(f'##cfg:{cfg}')
        if 'lr' in cfg['control']:
            cfg['douban_lr'] = float(cfg['control']['lr'])
        if 'milestones' in cfg['control']:
            cfg['douban_milestones'] = [int(_) for _ in cfg['control']['milestones'].split('-')]
        if 'factor' in cfg['control']:
            cfg['douban_factor'] = float(cfg['control']['factor'])


        if cfg['train_mode'] == 'fedavg':
            if cfg['num_nodes'] == 1:
                pass
            #     # simulate joint situation
            #     cfg['client'][model_name]['fraction'] = 1
            #     cfg['client'][model_name]['local_epoch'] = 1
            #     batch_size = {
            #         'user': {
            #             'Douban': 10,
            #         },
            #     }

            #     cfg['client'][model_name]['optimizer_name'] = 'SGD'
            #     cfg['client'][model_name]['lr'] = 0.01
            #     cfg['client'][model_name]['scheduler_name'] = 'CosineAnnealingLR'

                # cfg['client'][model_name]['optimizer_name'] = 'Adam'
                # cfg['client'][model_name]['lr'] = 1e-5
                # cfg['client'][model_name]['scheduler_name'] = 'None'
            else:
                batch_size = {
                    'user': {
                        'ML100K': 5, 
                        'ML1M': 10, 
                        'ML10M': 10, 
                        'ML20M': 10, 
                        'Douban': 10,
                        'taobaoclicksmall': 10,
                        'taobaoclickmedium': 10,
                        'taobaoclicklarge': 10
                    },
                    'item': {
                        'ML100K': 5, 
                        'ML1M': 10, 
                        'ML10M': 10, 
                        'ML20M': 10, 
                        'Douban': 10,
                        'taobaoclicksmall': 10,
                        'taobaoclickmedium': 10,
                        'taobaoclicklarge': 10
                    }
                }
                cfg['client'][model_name]['fraction'] = 0.1
                cfg['client'][model_name]['local_epoch'] = local_epoch
                cfg['client'][model_name]['optimizer_name'] = 'SGD'
                # cfg['client'][model_name]['lr'] = cfg['douban_lr']
                cfg['client'][model_name]['lr'] = 0.05
                # cfg['client'][model_name]['scheduler_name'] = 'CosineAnnealingLR'
                cfg['client'][model_name]['scheduler_name'] = 'MultiStepLR'
                cfg['client'][model_name]['factor'] = 0.1

                if cfg['num_nodes'] == 100:
                    cfg['client'][model_name]['milestones'] = [50,100,200]
                elif cfg['num_nodes'] == 300:
                    cfg['client'][model_name]['milestones'] = [100,200,400]
                elif cfg['num_nodes'] == 9999999:
                    cfg['client'][model_name]['milestones'] = [200,400,600]

                # cfg['client'][model_name]['milestones'] = cfg['douban_milestones']
                
        elif cfg['train_mode'] == 'joint':
            cfg['client'][model_name]['fraction'] = 1
            # cfg['client'][model_name]['optimizer_name'] = 'Adam'
            # cfg['client'][model_name]['lr'] = 1e-5
            # cfg['client'][model_name]['scheduler_name'] = 'None'

            cfg['client'][model_name]['optimizer_name'] = 'SGD'
            cfg['client'][model_name]['lr'] = 0.05
            # cfg['client'][model_name]['lr'] = 0.03
            # cfg['client'][model_name]['scheduler_name'] = 'CosineAnnealingLR'
            cfg['client'][model_name]['scheduler_name'] = 'MultiStepLR'
            
            cfg['client'][model_name]['factor'] = 0.1


            cfg['client'][model_name]['milestones'] = [30,100,200]
            # cfg['client'][model_name]['milestones'] = cfg['douban_milestones']
            # batch_size = {
            #     'user': {
            #         'Douban': 100,
            #     },
            # }

            # cfg['client'][model_name]['optimizer_name'] = 'SGD'
            # cfg['client'][model_name]['lr'] = 0.05
            # cfg['client'][model_name]['scheduler_name'] = 'MultiStepLR'
            # cfg['client'][model_name]['milestones'] = [20, 100]
            # cfg['client'][model_name]['factor'] = 0.1
            # cfg['client'][model_name]['scheduler_name'] = 'CosineAnnealingLR'


    cfg['client'][model_name]['momentum'] = 0.9
    cfg['client'][model_name]['nesterov'] = True
    cfg['client'][model_name]['betas'] = (0.9, 0.999)
    cfg['client'][model_name]['weight_decay'] = 5e-4
    cfg['client'][model_name]['batch_size'] = {'train': batch_size[cfg['data_mode']][cfg['data_name']],
                                     'test': batch_size[cfg['data_mode']][cfg['data_name']]}
    cfg['client'][model_name]['num_epochs'] = global_epoch

    cfg['server'][model_name]['optimizer_name'] = 'SGD'
    cfg['server'][model_name]['lr'] = 1
    cfg['server'][model_name]['momentum'] = 0
    cfg['server'][model_name]['weight_decay'] = 0
    cfg['server'][model_name]['nesterov'] = False

    print(f'--cfg:{cfg}')
    return

def fix_parameters(model):
    # print('fix_layers', cfg['fix_layers'], cfg['fix_layers'] == 'encoder')
    key_list = []
    for key,value in model.named_parameters():
        key_list.append(key)
    
    if cfg['fix_layers'] == 'last':
        for key,value in model.named_parameters():
            if key == key_list[-1] or key == key_list[-2]:
                value.requires_grad = True
            else:
                value.requires_grad = False
    elif cfg['fix_layers'] == 'encoder':
        for key,value in model.named_parameters():
            if 'encoder' in key:
                value.requres_grad = True
            else:
                value.requires_grad = False
    elif cfg['fix_layers'] == 'decoder':
        for key,value in model.named_parameters():
            if 'decoder' in key:
                value.requres_grad = True
            else:
                value.requires_grad = False

    return

def calculate_portion(num_average_nodes, num_active_nodes):
    partial_average_portion, non_partial_average_portion = None, None
    if cfg['weighted_average_method'] == 'None':
        partial_average_portion = 1/num_active_nodes
        non_partial_average_portion = 1/num_active_nodes
    elif cfg['weighted_average_method'] == 'squareroot':
        average_node_occupation = num_average_nodes / num_active_nodes
        weighted_average_node_occupation = average_node_occupation ** 0.5
        partial_average_portion = weighted_average_node_occupation / num_average_nodes
        non_partial_average_portion = (1-weighted_average_node_occupation) / (num_active_nodes-num_average_nodes)
    # print('partial_average_portion', partial_average_portion, non_partial_average_portion)
    return partial_average_portion, non_partial_average_portion



def init_final_layer(model):
    key_list = []
    for key,value in model.named_parameters():
        key_list.append(key)
    
    if cfg['fix_layers'] == 'last':
        for key,value in model.named_parameters():
            if key == key_list[-2] and 'weight' in key:
                nn.init.xavier_uniform_(value)        
            if key == key_list[-1] and 'bias' in key:
                value.data.zero_()    
    elif cfg['fix_layers'] == 'encoder':
        for key,value in model.named_parameters():
            split_key = key.split('.')
            if split_key[0] != 'encoder':
                continue
            if int(split_key[2])%3 == 0 and 'weight' in key:
                # print('gggggg', split_key)
                nn.init.xavier_uniform_(value)        
            elif int(split_key[2])%3 == 0 and 'bias' in key:
                value.data.zero_() 
    elif cfg['fix_layers'] == 'decoder':
        for key,value in model.named_parameters():
            split_key = key.split('.')
            if split_key[0] != 'decoder':
                continue
            if int(split_key[2])%3 == 0 and 'weight' in key:
                # print('gggggg', split_key)
                nn.init.xavier_uniform_(value)        
            elif int(split_key[2])%3 == 0 and 'bias' in key:
                value.data.zero_() 
    return



def make_stats():
    stats = {}
    stats_path = './res/stats'
    makedir_exist_ok(stats_path)
    filenames = os.listdir(stats_path)
    for filename in filenames:
        stats_name = os.path.splitext(filename)[0]
        stats[stats_name] = load(os.path.join(stats_path, filename))
    return stats


class Stats(object):
    def __init__(self, dim):
        self.dim = dim
        self.n_samples = 0
        self.n_features = None
        self.mean = None
        self.std = None

    def update(self, data):
        data = data.transpose(self.dim, -1).reshape(-1, data.size(self.dim))
        if self.n_samples == 0:
            self.n_samples = data.size(0)
            self.n_features = data.size(1)
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0)
        else:
            m = float(self.n_samples)
            n = data.size(0)
            new_mean = data.mean(dim=0)
            new_std = 0 if n == 1 else data.std(dim=0)
            old_mean = self.mean
            old_std = self.std
            self.mean = m / (m + n) * old_mean + n / (m + n) * new_mean
            self.std = torch.sqrt(m / (m + n) * old_std ** 2 + n / (m + n) * new_std ** 2 + m * n / (m + n) ** 2 * (
                    old_mean - new_mean) ** 2)
            self.n_samples += n
        return


def make_optimizer(model, model_name, tag, is_fine_tune=False):

    """
    Generate optimizer based on the parameters of model, the name of the model
    and the optimizer name

    Parameters:
        model - Object. The instance of model class
        model_name - str. The name of the model
        tag - str. Indicate if it is client or server

    Returns:
        optimizer - Object. The instance of corresponding optimizer class

    Raises:
        None
    """
    if not is_fine_tune:
        if cfg[tag][model_name]['optimizer_name'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=cfg[tag][model_name]['lr'], momentum=cfg[tag][model_name]['momentum'],
                                weight_decay=cfg[tag][model_name]['weight_decay'], nesterov=cfg[tag][model_name]['nesterov'])
        elif cfg[tag][model_name]['optimizer_name'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=cfg[tag][model_name]['lr'], betas=cfg[tag][model_name]['betas'],
                                weight_decay=cfg[tag][model_name]['weight_decay'])
        elif cfg[tag][model_name]['optimizer_name'] == 'LBFGS':
            optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag][model_name]['lr'])
        else:
            raise ValueError('Not valid optimizer name')

    # elif is_fine_tune:
    #     if cfg[tag][model_name]['optimizer_name'] == 'SGD':
    #         optimizer = optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
    #                             weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    #     elif cfg[tag]['optimizer_name'] == 'Adam':
    #         optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
    #                             weight_decay=cfg[tag]['weight_decay'])
    #     elif cfg[tag]['optimizer_name'] == 'LBFGS':
    #         optimizer = optim.LBFGS(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, model_name, tag):

    """
    Generate scheduler based on the optimizer, the name of the model
    and the scduler name. Scheduler would adjust the learning rate of optimizer.

    Parameters:
        optimizer - Object. The instance of optimizer class
        model_name - str. The name of the model
        tag - String. The name of the model

    Returns:j
        scheduler - Object. The instance of corresponding scheduler class

    Raises:
        None
    """

    # model_name = cfg['model_name']
    if cfg[tag][model_name]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag][model_name]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag][model_name]['step_size'], gamma=cfg[tag][model_name]['factor'])
    elif cfg[tag][model_name]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag][model_name]['milestones'],
                                                gamma=cfg[tag][model_name]['factor'])
    elif cfg[tag][model_name]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag][model_name]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag][model_name]['num_epochs'], eta_min=0)
    elif cfg[tag][model_name]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag][model_name]['factor'],
                                                        patience=cfg[tag][model_name]['patience'], verbose=False,
                                                        threshold=cfg[tag][model_name]['threshold'], threshold_mode='rel',
                                                        min_lr=cfg[tag][model_name]['min_lr'])
    elif cfg[tag][model_name]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag][model_name]['lr'], max_lr=10 * cfg[tag][model_name]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler
    # elif is_fine_tune:
    #     if cfg[tag]['scheduler_name'] == 'None':
    #         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    #     elif cfg[tag]['scheduler_name'] == 'StepLR':
    #         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    #     elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
    #         scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
    #                                                 gamma=cfg[tag]['factor'])
    #     elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
    #         scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    #     elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
    #         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['fine_tune_epoch'], eta_min=0)
    #     elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
    #         scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
    #                                                         patience=cfg[tag]['patience'], verbose=False,
    #                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
    #                                                         min_lr=cfg[tag]['min_lr'])
    #     elif cfg[tag]['scheduler_name'] == 'CyclicLR':
    #         scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg['fine_tune_lr'], max_lr=10 * cfg['fine_tune_lr'])
    #     else:
    #         raise ValueError('Not valid scheduler name')
    #     return scheduler


def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('../output/model/{}_{}.pt'.format(model_tag, load_tag)):
        result = load('../output/model/{}_{}.pt'.format(model_tag, load_tag))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = '../output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print('Resume from {}'.format(result['epoch']))
    return result


def collate(input):

    """
    for every key:value pair in input, concatenate the value(torch.tensor) (按行)

    Parameters:
        input - Dict. Input is the batch_size data that has been processed by 
            input_collate(batch), which is in data.py / input_collate(batch)

    Returns:
        input - Dict. Processed input dict. Since we have 1 dimension data in sub item.
            The final value of each key would be 1 long tensor, such as 8200 length. 

    Raises:
        None
    """

    for k in input:
        input[k] = torch.cat(input[k], 0)
    return input
