import os
import copy
import torch
import numpy as np
import models
import collections
from config import cfg
from scipy.sparse import csr_matrix
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from utils import collate, to_device

def split_dataset(dataset, num_nodes, data_split_mode):
    data_split = {}
    if data_split_mode == 'iid':
      
        data_split['train'], data_split_info = iid(dataset['train'], num_nodes)
        data_split['test'], _ = iid(dataset['test'], num_nodes)
    
    # elif 'non-iid' in cfg['data_split_mode']:
    #     data_split['train'], label_split = non_iid(dataset['train'], num_users)
    #     data_split['test'], _ = non_iid(dataset['test'], num_users, label_split)
    else:
        raise ValueError('Not valid data split mode')
    # print(data_split['test'])
    return data_split, data_split_info


def iid(dataset, num_nodes):
    if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M']:
        pass
    else:
        raise ValueError('Not valid data name')
    
    user_per_node = int(cfg['num_users']['data'] / num_nodes)
    data_split, idx = {}, list(range(cfg['num_users']['data']))
    data_split_info = collections.defaultdict(dict)
    
    for i in range(num_nodes):
        user_per_node_i = min(len(idx), user_per_node)
        data_split[i] = torch.tensor(idx)[torch.randperm(len(idx))[:user_per_node_i]].tolist()
        data_split_info[i]['num_users'] = user_per_node_i
        data_split_info[i]['num_items'] = cfg['num_items']['data']
        idx = list(set(idx) - set(data_split[i]))
    
    
    # for i in range(len(idx)): 
    #     data_split[i].append(idx[i])
    #     data_split_info[i]['num_users'] += 1
    # print(data_split_info)
    # print('gff', data_split_info[0])
    # print('wudi', data_split_info)
    return data_split, data_split_info


# def non_iid(dataset, num_users, label_split=None):
#     label = np.array(dataset.target)
#     cfg['non-iid-n'] = int(cfg['data_split_mode'].split('-')[-1])
#     shard_per_user = cfg['non-iid-n']
#     data_split = {i: [] for i in range(num_users)}

#     label_idx_split = collections.defaultdict(list)
#     for i in range(len(label)):
#         label_i = label[i].item()
#         label_idx_split[label_i].append(i)

#     shard_per_class = int(shard_per_user * num_users / cfg['classes_size'])
#     for label_i in label_idx_split:
#         label_idx = label_idx_split[label_i]
#         num_leftover = len(label_idx) % shard_per_class
#         leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
#         new_label_idx = np.array(label_idx[:-num_leftover]) if num_leftover > 0 else np.array(label_idx)
#         new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()
#         for i, leftover_label_idx in enumerate(leftover):
#             new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
#         label_idx_split[label_i] = new_label_idx
#     if label_split is None:
#         label_split = list(range(cfg['classes_size'])) * shard_per_class
#         label_split = torch.tensor(label_split)[torch.randperm(len(label_split))].tolist()
#         label_split = np.array(label_split).reshape((num_users, -1)).tolist()
#         for i in range(len(label_split)):
#             label_split[i] = np.unique(label_split[i]).tolist()
#     for i in range(num_users):
#         for label_i in label_split[i]:
#             idx = torch.arange(len(label_idx_split[label_i]))[torch.randperm(len(label_idx_split[label_i]))[0]].item()
#             data_split[i].extend(label_idx_split[label_i].pop(idx))
#     return data_split, label_split


class SplitDataset(Dataset):
    def __init__(self, dataset, idx):
        super().__init__()
        self.dataset = dataset
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        input = self.dataset[self.idx[index]]
        return input


class BatchDataset(Dataset):
    def __init__(self, dataset, seq_length):
        super().__init__()
        self.dataset = dataset
        self.seq_length = seq_length
        self.S = dataset[0]['label'].size(0)
        self.idx = list(range(0, self.S, seq_length))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        seq_length = min(self.seq_length, self.S - index)
        input = {'label': self.dataset[:]['label'][:, self.idx[index]:self.idx[index] + seq_length]}
        return input


def fetch_dataset(data_name, model_name=None, verbose=True):
    import datasets

    """
    1. Initialize the dataset class
    2. add transform attribute to the dataset class instance

    Parameters:
        data_name - String. The name of chosen dataset as well as the class name.
            For example: ML100K

    Returns:
        dataset - Dict. dataset['train'] is the train dataset instance. dataset['test']
            is the test dataset instance

    Raises:
        None
    """

    model_name = cfg['model_name'] if model_name is None else model_name
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    
    root = os.path.join('data', '{}'.format(data_name))
    # root = './data/{}'.format(data_name)
    if data_name in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'taobaoclicksmall', 'taobaoclickmedium', 'taobaoclicklarge']:
        if data_name in ['taobaoclicksmall', 'taobaoclickmedium', 'taobaoclicklarge']:
            root = os.path.join('data', 'taobaoclick')
        # initialize the corresponding class of data_name in datasets / movielens.py
        # put the corresponding class instance in dataset['train']
        # put the corresponding class instance in dataset['test']
        # if data_name == 'taobaoclicksmall':
        #     dataset['train'] = eval(
        #         'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"], '
        #         'target_mode=cfg["target_mode"])'.format(data_name))
        #     dataset['test'] = eval(
        #         'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"], '
        #         'target_mode=cfg["target_mode"], )'.format(data_name))
        # elif data_name == 'taobaoclickmedium':
        #     dataset['train'] = eval(
        #         'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"], '
        #         'target_mode=cfg["target_mode"])'.format(data_name))
        #     dataset['test'] = eval(
        #         'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"], '
        #         'target_mode=cfg["target_mode"])'.format(data_name))
        # elif data_name == 'taobaoclicklarge':
        #     dataset['train'] = eval(
        #         'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"], '
        #         'target_mode=cfg["target_mode"])'.format(data_name))
        #     dataset['test'] = eval(
        #         'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"], '
        #         'target_mode=cfg["target_mode"])'.format(data_name))
        # else:
        dataset['train'] = eval(
            'datasets.{}(root=root, split=\'train\', data_mode=cfg["data_mode"], '
            'target_mode=cfg["target_mode"])'.format(data_name))
        dataset['test'] = eval(
            'datasets.{}(root=root, split=\'test\', data_mode=cfg["data_mode"], '
            'target_mode=cfg["target_mode"])'.format(data_name))
        
        # for index in range(cfg['unique_user_num']):
        #     if index in cfg['test_user_unique']:  
        #         print('train', dataset['train'][index]['target_rating'])
        #         print('test', dataset['test'][index]['target_rating'])

        if model_name in ['base', 'mf', 'gmf', 'mlp', 'nmf']:
            # add transform attribute to corresponding class instance
            dataset = make_pair_transform(dataset)
        elif model_name in ['ae']:
            # add transform attribute to corresponding class instance
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

    """
    Define a batch data handler.

    Parameters:
        batch - List. The list result returned by the __getitem()__ of the dataset instance. 

    Returns:
        self-defined dict or default_collate(batch)

    Raises:
        None
    """

    if isinstance(batch[0], dict):
        
        # res = collections.defaultdict(list)
        # for key in batch[0]:
        #     for item in batch:
        #         res[key].append(item[key])

        res = {key: [b[key] for b in batch] for key in batch[0]}
        return res
    else:
        return default_collate(batch)



def make_data_loader(dataset, batch_size=None, shuffle=None, sampler=None):

    """
    generate DataLoader instance(iterable) based on default parameters setting and passing parameters

    Parameters:
        dataset - Dict. dataset['train'] is the train dataset instance. dataset['test']
            is the test dataset instance

    Returns:
        data_loader - Dict. data_loader['train'] is the train DataLoader instance.
            data_loader['test'] is the test DataLoader instance. The DataLoader instance is
            iterable (可迭代对象).

    Raises:
        None
    """

    data_loader = {}
    cur_model = cfg['model_name']
    # iterate dataset['train'] and dataset['test']
    for k in dataset:
        # if we dont pass batch_size parameter, use default parameter in cfg
        # default parameter is defined in utils.py / process_control()
        _batch_size = cfg[cur_model]['batch_size'][k] if batch_size is None else batch_size[k]
        if cfg['train_mode'] == 'fedsgd':
            _batch_size = int(cfg[cfg['model_name']]['fraction'] * cfg['num_users']['data'])
        if cfg['train_mode'] == 'fedavg' and cfg['control']['num_nodes'] == 'max':
            _batch_size = 1
        # if we dont pass shuffle parameter, use default parameter in cfg
        _shuffle = cfg[cur_model]['shuffle'][k] if shuffle is None else shuffle[k]
        # if the cfg['device'] is cuda, we set pin_memory to True
        _pin_memory = True if cfg['device'] == 'cuda' else False

        # if we dont pass sampler
        if sampler is None:
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, shuffle=_shuffle,
                                        pin_memory=_pin_memory, num_workers=cfg['num_workers'], collate_fn=input_collate,
                                        worker_init_fn=np.random.seed(cfg['seed']))
        else:
            # if we pass sampler
            data_loader[k] = DataLoader(dataset=dataset[k], batch_size=_batch_size, sampler=sampler[k],
                                        pin_memory=_pin_memory, num_workers=cfg['num_workers'], collate_fn=input_collate,
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
        # elif self.data_mode == 'item':
        #     input['item'] = input['item'].repeat(input['user'].size(0))
        #     input['target_item'] = input['target_item'].repeat(input['target_user'].size(0))
        #     if self.info == 1:
        #         if 'item_attr' in input:
        #             input['item_attr'] = input['item_attr'].view(1, -1).repeat(input['user'].size(0), 1)
        #         if 'target_item_attr' in input:
        #             input['target_item_attr'] = input['target_item_attr'].view(1, -1).repeat(
        #                 input['target_user'].size(0), 1)
        #     else:
        #         if 'user_profile' in input:
        #             del input['user_profile']
        #         if 'target_user_profile' in input:
        #             del input['target_user_profile']
        #         if 'item_attr' in input:
        #             del input['item_attr']
        #         if 'target_item_attr' in input:
        #             del input['target_item_attr']
        else:
            raise ValueError('Not valid data mode')
        return input


class FlatInput(torch.nn.Module):

    """
    Modify input data
   
    Parameters:
        data_mode - String. cfg['data_mode'] => user or item.
        info - Integer. cfg['info'] => 1 or 0.
        num_users - Integer. The number of unique users.
        num_items - Integer. The number of unique items.

    Returns:
        input - Dict. Processed input data

    Raises:
        None
    """

    def __init__(self, data_mode, info, num_users, num_items):
        super().__init__()
        self.data_mode = data_mode
        self.info = info
        self.num_users = num_users
        self.num_items = num_items

    def forward(self, input):
        if self.data_mode == 'user':
            # copy the single tensor for input['item'].size(0) times
            # For example, tensor([597]) and input['item'].size(0) == 24
            # tensor([597, 597.....])
            input['user'] = input['user'].repeat(input['item'].size(0))
            input['target_user'] = input['target_user'].repeat(input['target_item'].size(0))
            if self.info == 1:
                if 'user_profile' in input:
                    # reshape to 1 row and x col
                    input['user_profile'] = input['user_profile'].view(1, -1)
                    if input['item'].size(0) == 0 and input['target_item'].size(0) == 0:
                        input['user_profile'] = input['user_profile'].repeat(input['item'].size(0), 1)
                if 'item_attr' in input:
                    input['item_attr'] = input['item_attr'].sum(dim=0, keepdim=True)
                    if input['item'].size(0) == 0 and input['target_item'].size(0) == 0:
                        input['item_attr'] = input['item_attr'].repeat(input['item'].size(0), 1)
                # delete side information in target
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
            else:
                # delete side information if self.info != 1
                if 'user_profile' in input:
                    del input['user_profile']
                if 'target_user_profile' in input:
                    del input['target_user_profile']
                if 'item_attr' in input:
                    del input['item_attr']
                if 'target_item_attr' in input:
                    del input['target_item_attr']
        # elif self.data_mode == 'item':
        #     input['item'] = input['item'].repeat(input['user'].size(0))
        #     input['target_item'] = input['target_item'].repeat(input['target_user'].size(0))
        #     if self.info == 1:
        #         if 'user_profile' in input:
        #             input['user_profile'] = input['user_profile'].sum(dim=0, keepdim=True)
        #             if input['user'].size(0) == 0 and input['target_user'].size(0) == 0:
        #                 input['user_profile'] = input['user_profile'].repeat(input['user'].size(0), 1)
        #         if 'item_attr' in input:
        #             input['item_attr'] = input['item_attr'].view(1, -1)
        #             if input['user'].size(0) == 0 and input['target_user'].size(0) == 0:
        #                 input['item_attr'] = input['item_attr'].repeat(input['user'].size(0), 1)
        #         if 'target_user_profile' in input:
        #             del input['target_user_profile']
        #         if 'target_item_attr' in input:
        #             del input['target_item_attr']
        #     else:
        #         if 'user_profile' in input:
        #             del input['user_profile']
        #         if 'target_user_profile' in input:
        #             del input['target_user_profile']
        #         if 'item_attr' in input:
        #             del input['item_attr']
        #         if 'target_item_attr' in input:
        #             del input['target_item_attr']
        else:
            raise ValueError('Not valid data mode')
        return input

