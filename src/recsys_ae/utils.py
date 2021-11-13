import collections.abc as container_abcs
import errno
import numpy as np
import os
import pickle
import torch
import torch.optim as optim
from itertools import repeat
from torchvision.utils import save_image
from config import cfg


def check_exists(path):
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


def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
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
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output


def process_dataset(dataset):
    cfg['data_size'] = {'train': len(dataset['train']), 'test': len(dataset['test'])}
    cfg['num_users'], cfg['num_items'] = dataset['train'].num_users, dataset['train'].num_items
    if cfg['info'] == 1:
        cfg['info_size'] = {}
        if hasattr(dataset['train'], 'user_profile'):
            cfg['info_size']['user_profile'] = dataset['train'].user_profile['data'].shape[1]
        if hasattr(dataset['train'], 'item_attr'):
            cfg['info_size']['item_attr'] = dataset['train'].item_attr['data'].shape[1]
    else:
        cfg['info_size'] = None
    return


def process_control():
    cfg['data_name'] = cfg['control']['data_name']
    cfg['data_mode'] = cfg['control']['data_mode']
    cfg['target_mode'] = cfg['control']['target_mode']
    cfg['model_name'] = cfg['control']['model_name']
    cfg['info'] = float(cfg['control']['info']) if 'info' in cfg['control'] else 0
    if 'data_split_mode' in cfg['control']:
        cfg['data_split_mode'] = cfg['control']['data_split_mode']
        if 'genre' in cfg['data_split_mode']:
            if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
                cfg['num_organizations'] = 18
            else:
                raise ValueError('Not valid data name')
        elif 'random' in cfg['data_split_mode']:
            cfg['num_organizations'] = int(cfg['data_split_mode'].split('-')[1])
        else:
            raise ValueError('Not valid data split mode')
    cfg['assist'] = {}
    if 'ar' in cfg['control']:
        ar_list = cfg['control']['ar'].split('-')
        cfg['assist']['ar_mode'] = ar_list[0]
        cfg['assist']['ar'] = float(ar_list[1])
    if 'aw' in cfg['control']:
        cfg['assist']['aw_mode'] = cfg['control']['aw']
    if 'match_rate' in cfg['control']:
        cfg['assist']['match_rate'] = float(cfg['control']['match_rate'])
    cfg['base'] = {}
    cfg['mf'] = {'hidden_size': 128}
    cfg['gmf'] = {'hidden_size': 128}
    cfg['mlp'] = {'hidden_size': [128, 64, 32, 16]}
    cfg['nmf'] = {'hidden_size': [128, 64, 32, 16]}
    cfg['ae'] = {'encoder_hidden_size': [256, 128], 'decoder_hidden_size': [128, 256]}
    batch_size = {'user': {'ML100K': 100, 'ML1M': 500, 'ML10M': 5000, 'ML20M': 5000, 'NFP': 5000},
                  'item': {'ML100K': 100, 'ML1M': 500, 'ML10M': 1000, 'ML20M': 1000, 'NFP': 1000}}
    model_name = cfg['model_name']
    cfg[model_name]['shuffle'] = {'train': True, 'test': False}
    cfg[model_name]['optimizer_name'] = 'Adam'
    cfg[model_name]['lr'] = 1e-3
    cfg[model_name]['momentum'] = 0.9
    cfg[model_name]['nesterov'] = True
    cfg[model_name]['betas'] = (0.9, 0.999)
    cfg[model_name]['weight_decay'] = 5e-4
    cfg[model_name]['scheduler_name'] = 'None'
    cfg[model_name]['batch_size'] = {'train': batch_size[cfg['data_mode']][cfg['data_name']],
                                     'test': batch_size[cfg['data_mode']][cfg['data_name']]}
    cfg[model_name]['num_epochs'] = 200 if model_name != 'base' else 1
    cfg['local'] = {}
    cfg['local']['shuffle'] = {'train': True, 'test': False}
    cfg['local']['optimizer_name'] = 'Adam'
    cfg['local']['lr'] = 1e-3
    cfg['local']['momentum'] = 0.9
    cfg['local']['nesterov'] = True
    cfg['local']['betas'] = (0.9, 0.999)
    cfg['local']['weight_decay'] = 5e-4
    cfg['local']['scheduler_name'] = 'None'
    cfg['local']['batch_size'] = {'train': batch_size[cfg['data_mode']][cfg['data_name']],
                                  'test': batch_size[cfg['data_mode']][cfg['data_name']]}
    cfg['local']['num_epochs'] = 20
    cfg['global'] = {}
    cfg['global']['num_epochs'] = 10
    cfg['assist']['optimizer_name'] = 'LBFGS'
    cfg['assist']['lr'] = 1
    cfg['assist']['num_epochs'] = 10
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


def make_optimizer(model, tag):
    if cfg[tag]['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=cfg[tag]['lr'], momentum=cfg[tag]['momentum'],
                              weight_decay=cfg[tag]['weight_decay'], nesterov=cfg[tag]['nesterov'])
    elif cfg[tag]['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg[tag]['lr'], betas=cfg[tag]['betas'],
                               weight_decay=cfg[tag]['weight_decay'])
    elif cfg[tag]['optimizer_name'] == 'LBFGS':
        optimizer = optim.LBFGS(model.parameters(), lr=cfg[tag]['lr'])
    else:
        raise ValueError('Not valid optimizer name')
    return optimizer


def make_scheduler(optimizer, tag):
    if cfg[tag]['scheduler_name'] == 'None':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[65535])
    elif cfg[tag]['scheduler_name'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg[tag]['step_size'], gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'MultiStepLR':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg[tag]['milestones'],
                                                   gamma=cfg[tag]['factor'])
    elif cfg[tag]['scheduler_name'] == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    elif cfg[tag]['scheduler_name'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg[tag]['num_epochs'], eta_min=0)
    elif cfg[tag]['scheduler_name'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg[tag]['factor'],
                                                         patience=cfg[tag]['patience'], verbose=False,
                                                         threshold=cfg[tag]['threshold'], threshold_mode='rel',
                                                         min_lr=cfg[tag]['min_lr'])
    elif cfg[tag]['scheduler_name'] == 'CyclicLR':
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg[tag]['lr'], max_lr=10 * cfg[tag]['lr'])
    else:
        raise ValueError('Not valid scheduler name')
    return scheduler


def resume(model_tag, load_tag='checkpoint', verbose=True):
    if os.path.exists('./output/model/{}_{}.pt'.format(model_tag, load_tag)):
        result = load('./output/model/{}_{}.pt'.format(model_tag, load_tag))
    else:
        print('Not exists model tag: {}, start from scratch'.format(model_tag))
        from datetime import datetime
        from logger import Logger
        last_epoch = 1
        logger_path = 'output/runs/train_{}_{}'.format(cfg['model_tag'], datetime.now().strftime('%b%d_%H-%M-%S'))
        logger = Logger(logger_path)
        result = {'epoch': last_epoch, 'logger': logger}
    if verbose:
        print('Resume from {}'.format(result['epoch']))
    return result


def collate(input):
    for k in input:
        input[k] = torch.cat(input[k], 0)
    return input
