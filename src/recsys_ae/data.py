import copy
import torch
import numpy as np
import models
from config import cfg
from scipy.sparse import csr_matrix
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device


def fetch_dataset(data_name, model_name=None, verbose=True):
    import datasets

    model_name = cfg['model_name'] if model_name is None else model_name
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
        dataset['train'] = eval(
            'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"], '
            'target_mode=cfg["target_mode"])'.format(data_name))
        dataset['test'] = eval(
            'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"], '
            'target_mode=cfg["target_mode"])'.format(data_name))
        if model_name in ['base', 'mf', 'gmf', 'mlp', 'nmf']:
            dataset = make_pair_transform(dataset)
        elif model_name in ['ae']:
            dataset = make_flat_transform(dataset)
        else:
            raise ValueError('Not valid model name')
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def make_pair_transform(dataset):
    import datasets
    if 'train' in dataset:
        dataset['train'].transform = datasets.Compose([PairInput(cfg['data_mode'], cfg['info'])])
    if 'test' in dataset:
        dataset['test'].transform = datasets.Compose([PairInput(cfg['data_mode'], cfg['info'])])
    return dataset


def make_flat_transform(dataset):
    import datasets
    if 'train' in dataset:
        dataset['train'].transform = datasets.Compose(
            [FlatInput(cfg['data_mode'], cfg['info'], dataset['train'].num_users, dataset['train'].num_items)])
    if 'test' in dataset:
        dataset['test'].transform = datasets.Compose(
            [FlatInput(cfg['data_mode'], cfg['info'], dataset['test'].num_users, dataset['test'].num_items)])
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        return {key: [b[key] for b in batch] for key in batch[0]}
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, batch_size=None, shuffle=None, sampler=None):
    data_loader = {}
    for k in dataset:
        _batch_size = cfg[tag]['batch_size'][k] if batch_size is None else batch_size[k]
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=True, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


class PairInput(torch.nn.Module):
    def __init__(self, data_mode, info):
        super().__init__()
        self.data_mode = data_mode
        self.info = info

    def forward(self, input):
        if self.data_mode == 'user':
            input['user'] = input['user'].repeat(input['item'].size(0))
            input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
            if self.info == 1:
                if 'user_profile' in input:
                    input['user_profile'] = input['user_profile'].view(1, -1).repeat(input['item'].size(0), 1)
                if 'target_user_profile' in input:
                    input['target_user_profile'] = input['target_user_profile'].view(1, -1).repeat(
                        input['target_item'].size(0), 1)
            else:
                if 'user_profile' in input:
                    del input['user_profile']
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'item_attr' in input:
                    del input['item_attr']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
        elif self.data_mode == 'item':
            input['item'] = input['item'].repeat(input['user'].size(0))
            input['target_item'] = input['target_item'].repeat(input['target_user'].size(0))
            if self.info == 1:
                if 'item_attr' in input:
                    input['item_attr'] = input['item_attr'].view(1, -1).repeat(input['user'].size(0), 1)
                if 'target_item_attr' in input:
                    input['target_item_attr'] = input['target_item_attr'].view(1, -1).repeat(
                        input['target_user'].size(0), 1)
            else:
                if 'user_profile' in input:
                    del input['user_profile']
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'item_attr' in input:
                    del input['item_attr']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
        else:
            raise ValueError('Not valid data mode')
        return input


class FlatInput(torch.nn.Module):
    def __init__(self, data_mode, info, num_users, num_items):
        super().__init__()
        self.data_mode = data_mode
        self.info = info
        self.num_users = num_users
        self.num_items = num_items

    def forward(self, input):
        if self.data_mode == 'user':
            input['user'] = input['user'].repeat(input['item'].size(0))
            input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
            if self.info == 1:
                if 'user_profile' in input:
                    input['user_profile'] = input['user_profile'].view(1, -1)
                    if input['item'].size(0) == 0 and input['target_item'].size(0) == 0:
                        input['user_profile'] = input['user_profile'].repeat(input['item'].size(0), 1)
                if 'item_attr' in input:
                    input['item_attr'] = input['item_attr'].sum(dim=0, keepdim=True)
                    if input['item'].size(0) == 0 and input['target_item'].size(0) == 0:
                        input['item_attr'] = input['item_attr'].repeat(input['item'].size(0), 1)
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
            else:
                if 'user_profile' in input:
                    del input['user_profile']
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'item_attr' in input:
                    del input['item_attr']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
        elif self.data_mode == 'item':
            input['item'] = input['item'].repeat(input['user'].size(0))
            input['target_item'] = input['target_item'].repeat(input['target_user'].size(0))
            if self.info == 1:
                if 'user_profile' in input:
                    input['user_profile'] = input['user_profile'].sum(dim=0, keepdim=True)
                    if input['user'].size(0) == 0 and input['target_user'].size(0) == 0:
                        input['user_profile'] = input['user_profile'].repeat(input['user'].size(0), 1)
                if 'item_attr' in input:
                    input['item_attr'] = input['item_attr'].view(1, -1)
                    if input['user'].size(0) == 0 and input['target_user'].size(0) == 0:
                        input['item_attr'] = input['item_attr'].repeat(input['user'].size(0), 1)
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
            else:
                if 'user_profile' in input:
                    del input['user_profile']
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'item_attr' in input:
                    del input['item_attr']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
        else:
            raise ValueError('Not valid data mode')
        return input


def split_dataset(dataset):
    if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
        if 'genre' in cfg['data_split_mode']:
            if cfg['data_mode'] == 'user':
                num_organizations = cfg['num_organizations']
                item_attr = torch.tensor(dataset['train'].item_attr['data'])
                zero_mask = torch.tensor(dataset['train'].item_attr['data']).sum(dim=-1) == 0
                item_attr[zero_mask] = 1
                all_filled = False
                while not all_filled:
                    all_filled = True
                    data_split = []
                    data_split_idx = torch.multinomial(item_attr, 1).view(-1).numpy()
                    for i in range(num_organizations):
                        data_split_i = np.where(data_split_idx == i)[0]
                        data_split.append(torch.tensor(data_split_i))
                        if len(data_split_i) == 0 or len(dataset['train'].data[:, data_split_i].data) == 0 or len(
                                dataset['test'].data[:, data_split_i].data) == 0 or len(
                            dataset['train'].target[:, data_split_i].data) == 0 or len(
                            dataset['test'].target[:, data_split_i].data) == 0:
                            all_filled = False
            elif cfg['data_mode'] == 'item':
                raise NotImplementedError
            else:
                raise ValueError('Not valid data mode')
        elif 'random' in cfg['data_split_mode']:
            if cfg['data_mode'] == 'user':
                num_items = dataset['train'].num_items['data']
                num_organizations = cfg['num_organizations']
                data_split = list(torch.randperm(num_items).split(num_items // num_organizations))
                data_split = data_split[:num_organizations - 1] + [torch.cat(data_split[num_organizations - 1:])]
            elif cfg['data_mode'] == 'item':
                num_users = dataset['train'].num_users['data']
                num_organizations = cfg['num_organizations']
                data_split = list(torch.randperm(num_users).split(num_users // num_organizations))
                data_split = data_split[:num_organizations - 1] + [torch.cat(data_split[num_organizations - 1:])]
            else:
                raise ValueError('Not valid data mode')
        else:
            raise ValueError('Not valid data split mode')
    else:
        raise ValueError('Not valid data name')
    return data_split


def make_split_dataset(data_split):
    num_organizations = len(data_split)
    dataset = []
    for i in range(num_organizations):
        data_split_i = data_split[i]
        dataset_i = fetch_dataset(cfg['data_name'], model_name=cfg['model_name'], verbose=False)
        for k in dataset_i:
            dataset_i[k].data = dataset_i[k].data[:, data_split_i]
            dataset_i[k].target = dataset_i[k].target[:, data_split_i]
            if cfg['data_mode'] == 'user':
                if hasattr(dataset_i[k], 'item_attr'):
                    shape = (-1, dataset_i[k].item_attr['data'][data_split_i].shape[-1])
                    dataset_i[k].item_attr['data'] = dataset_i[k].item_attr['data'][data_split_i].reshape(shape)
                    shape = (-1, dataset_i[k].item_attr['target'][data_split_i].shape[-1])
                    dataset_i[k].item_attr['target'] = dataset_i[k].item_attr['target'][data_split_i].reshape(shape)
            elif cfg['data_mode'] == 'item':
                if hasattr(dataset_i[k], 'user_profile'):
                    shape = (-1, dataset_i[k].user_profile['data'][data_split_i].shape[-1])
                    dataset_i[k].user_profile['data'] = dataset_i[k].user_profile['data'][data_split_i].reshape(shape)
                    shape = (-1, dataset_i[k].user_profile['target'][data_split_i].shape[-1])
                    dataset_i[k].user_profile['target'] = dataset_i[k].user_profile['target'][data_split_i].reshape(
                        shape)
            else:
                raise ValueError('Not valid data mode')
        if cfg['model_name'] in ['base', 'mf', 'gmf', 'mlp', 'nmf']:
            dataset_i = make_pair_transform(dataset_i)
        elif cfg['model_name'] in ['ae']:
            dataset_i = make_flat_transform(dataset_i)
        dataset.append(dataset_i)
    return dataset
