import numpy as np
import scipy
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .datasets_utils import download_url, extract_file
from scipy.sparse import csr_matrix
from config import cfg

class Anime(Dataset): 
    data_name = 'Anime'
    filename = 'anime_archive.zip'  # https://www.kaggle.com/datasets/fengzhujoey/douban-datasetratingreviewside-information

    def __init__(self, root, split, data_mode, target_mode, transform=None):
        self.sliced_user_count = 6000
        self.root = os.path.expanduser(root)
        self.split = split
        self.data_mode = data_mode
        self.target_mode = target_mode
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.data, self.target = load(os.path.join(self.processed_folder, self.target_mode, '{}.pt'.format(self.split)),
                                      mode='pickle')

        self.user_profile = {}
        if self.data_mode == 'user':
            pass
        else:
            raise ValueError('Not valid data mode')

    def __getitem__(self, index):
        data = self.data[index].tocoo()
        target = self.target[index].tocoo()
        if self.data_mode == 'user':
            input = {'user': torch.tensor(np.array([index]), dtype=torch.long),
                     'item': torch.tensor(data.col, dtype=torch.long),
                     'rating': torch.tensor(data.data),
                     'target_user': torch.tensor(np.array([index]), dtype=torch.long),
                     'target_item': torch.tensor(target.col, dtype=torch.long),
                     'target_rating': torch.tensor(target.data)}
            if 'data' in self.user_profile:
                input['user_profile'] = torch.tensor(self.user_profile['data'][index])
            if 'target' in self.user_profile:
                input['target_user_profile'] = torch.tensor(self.user_profile['target'][index])
        elif self.data_mode == 'item':
            input = {'user': torch.tensor(data.col, dtype=torch.long),
                     'item': torch.tensor(np.array([index]), dtype=torch.long),
                     'rating': torch.tensor(data.data),
                     'target_user': torch.tensor(target.col, dtype=torch.long),
                     'target_item': torch.tensor(np.array([index]), dtype=torch.long),
                     'target_rating': torch.tensor(target.data)}
            if 'data' in self.user_profile:
                input['user_profile'] = torch.tensor(self.user_profile['data'][data.col])
            if 'target' in self.user_profile:
                input['target_user_profile'] = torch.tensor(self.user_profile['target'][target.col])
            if 'data' in self.item_attr:
                input['item_attr'] = torch.tensor(self.item_attr['data'][index])
            if 'target' in self.item_attr:
                input['target_item_attr'] = torch.tensor(self.item_attr['target'][index])
        else:
            raise ValueError('Not valid data mode')
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        if self.data_mode == 'user':
            len_ = self.num_users['data']
        elif self.data_mode == 'item':
            len_ = self.num_items['data']
        else:
            raise ValueError('Not valid data mode')
        return len_

    @property
    def num_users(self):
        if self.data_mode == 'user':
            num_users_ = {'data': self.data.shape[0], 'target': self.target.shape[0]}
        elif self.data_mode == 'item':
            num_users_ = {'data': self.data.shape[1], 'target': self.target.shape[1]}
        else:
            raise ValueError('Not valid data mode')
        return num_users_

    @property
    def num_items(self):
        if self.data_mode == 'user':
            num_items_ = {'data': self.data.shape[1], 'target': self.target.shape[1]}
        elif self.data_mode == 'item':
            num_items_ = {'data': self.data.shape[0], 'target': self.target.shape[0]}
        else:
            raise ValueError('Not valid data mode')
        return num_items_

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()
        extract_file(os.path.join(self.raw_folder, self.filename))
        train_set, test_set = self.make_explicit_data()
        save(train_set, os.path.join(self.processed_folder, 'explicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'explicit', 'test.pt'), mode='pickle')
        train_set, test_set = self.make_implicit_data()
        save(train_set, os.path.join(self.processed_folder, 'implicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'implicit', 'test.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        user = []
        item = []
        rating = []

        data_i = pd.read_csv(os.path.join(self.raw_folder, 'rating.csv'), delimiter=',')
        data_i = data_i.drop(data_i[data_i['rating'] == -1].index)

        user_i = data_i.iloc[:, 0].to_numpy()       
        item_i = data_i.iloc[:, 1].to_numpy()
        item_id_i, item_inv_i = np.unique(item_i, return_inverse=True)
        item_id_map_i = {item_id_i[i]: i for i in range(len(item_id_i))}
        item_i = np.array([item_id_map_i[i] for i in item_id_i], dtype=np.int64)[item_inv_i].reshape(item_i.shape)
        rating_i = data_i.iloc[:, 2].astype(np.float32)
        user.append(user_i)

        item.append(item_i)
        rating.append(rating_i)
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        rating = np.concatenate(rating, axis=0)

        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)

        data = csr_matrix((rating, (user, item)), shape=(M, N))
        nonzero_user, nonzero_item = data.nonzero()
        _, count_nonzero_user = np.unique(nonzero_user, return_counts=True)
        _, count_nonzero_item = np.unique(nonzero_item, return_counts=True)

        dense_user_mask = count_nonzero_user >= 20
        dense_item_mask = count_nonzero_item >= 20
        dense_user_id = np.arange(len(user_id))[dense_user_mask]
        dense_item_id = np.arange(len(item_id))[dense_item_mask]
        dense_mask = np.logical_and(np.isin(user, dense_user_id), np.isin(item, dense_item_id))
        user = user[dense_mask]
        item = item[dense_mask]
        rating = rating[dense_mask]

        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)

        sliced_user = user_id[:self.sliced_user_count]
        sliced_item = item_id[:]
        sliced_mask = np.logical_and(np.isin(user, sliced_user), np.isin(item, sliced_item))

        user = user[sliced_mask]
        item = item[sliced_mask]
        rating = rating[sliced_mask]
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)  

        idx = np.random.permutation(user.shape[0])
        num_train = int(user.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        train_user, train_item, train_rating = user[train_idx], item[train_idx], rating[train_idx]
        test_user, test_item, test_rating = user[test_idx], item[test_idx], rating[test_idx]
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        train_target = train_data
        test_data = train_data
        test_target = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return (train_data, train_target), (test_data, test_target)

    def make_implicit_data(self):

        user = []
        item = []
        rating = []
        data_i = pd.read_csv(os.path.join(self.raw_folder, 'rating.csv'), delimiter=',')
        data_i = data_i.drop(data_i[data_i['rating'] == -1].index)
        user_i = data_i.iloc[:, 0].to_numpy()      
        item_i = data_i.iloc[:, 1].to_numpy()
        item_id_i, item_inv_i = np.unique(item_i, return_inverse=True)
        item_id_map_i = {item_id_i[i]: i for i in range(len(item_id_i))}
        item_i = np.array([item_id_map_i[i] for i in item_id_i], dtype=np.int64)[item_inv_i].reshape(item_i.shape)
        rating_i = data_i.iloc[:, 2].astype(np.float32)
        user.append(user_i)

        item.append(item_i)
        rating.append(rating_i)
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        rating = np.concatenate(rating, axis=0)

        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)

        data = csr_matrix((rating, (user, item)), shape=(M, N))
        nonzero_user, nonzero_item = data.nonzero()
        _, count_nonzero_user = np.unique(nonzero_user, return_counts=True)
        _, count_nonzero_item = np.unique(nonzero_item, return_counts=True)

        dense_user_mask = count_nonzero_user >= 20
        dense_item_mask = count_nonzero_item >= 20
        dense_user_id = np.arange(len(user_id))[dense_user_mask]
        dense_item_id = np.arange(len(item_id))[dense_item_mask]
        dense_mask = np.logical_and(np.isin(user, dense_user_id), np.isin(item, dense_item_id))
        user = user[dense_mask]
        item = item[dense_mask]
        rating = rating[dense_mask]

        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)

        sliced_user = user_id[:self.sliced_user_count]
        sliced_item = item_id[:]
        sliced_mask = np.logical_and(np.isin(user, sliced_user), np.isin(item, sliced_item))

        user = user[sliced_mask]
        item = item[sliced_mask]
        rating = rating[sliced_mask]
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)

        idx = np.random.permutation(user.shape[0])
        num_train = int(user.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        train_user, train_item, train_rating = user[train_idx], item[train_idx], rating[train_idx]
        train_rating[train_rating < 8] = 0
        train_rating[train_rating >= 8] = 1
        test_user, test_item, test_rating = user[test_idx], item[test_idx], rating[test_idx]
        test_rating[test_rating < 8] = 0
        test_rating[test_rating >= 8] = 1
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        train_target = train_data
        test_data = train_data
        test_target = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return (train_data, train_target), (test_data, test_target)
