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


class Amazon(Dataset):
    data_name = 'Amazon'
    # file = [('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/AMAZON_FASHION.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/All_Beauty.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Appliances.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Arts_Crafts_and_Sewing.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Automotive.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/CDs_and_Vinyl.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Cell_Phones_and_Accessories.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Clothing_Shoes_and_Jewelry.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Electronics.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Gift_Cards.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Grocery_and_Gourmet_Food.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Home_and_Kitchen.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Industrial_and_Scientific.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Kindle_Store.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Luxury_Beauty.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Magazine_Subscriptions.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Movies_and_TV.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Musical_Instruments.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Office_Products.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Patio_Lawn_and_Garden.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Pet_Supplies.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Prime_Pantry.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Software.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Sports_and_Outdoors.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Tools_and_Home_Improvement.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Toys_and_Games.csv', None),
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games.csv', None),
    #         ]
    file = [
            # ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Books.csv', None),
            # ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music.csv', None),
            ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Movies_and_TV.csv', None),
            # ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Video_Games.csv', None),
            ]
    # file = [
    #         ('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFilesSmall/Digital_Music.csv', None),

    #         ]
    # genre = ['AMAZON_FASHION', 'All_Beauty', 'Appliances', 'Arts_Crafts_and_Sewing', 'Automotive', 'Books',
    #          'CDs_and_Vinyl', 'Cell_Phones_and_Accessories', 'Clothing_Shoes_and_Jewelry', 'Digital_Music',
    #          'Electronics', 'Gift_Cards', 'Grocery_and_Gourmet_Food', 'Home_and_Kitchen',
    #          'Industrial_and_Scientific', 'Kindle_Store', 'Luxury_Beauty', 'Magazine_Subscriptions',
    #          'Movies_and_TV', 'Musical_Instruments', 'Office_Products', 'Patio_Lawn_and_Garden', 'Pet_Supplies',
    #          'Prime_Pantry', 'Software', 'Sports_and_Outdoors', 'Tools_and_Home_Improvement', 'Toys_and_Games',
    #          'Video_Games']
    # genre = ['Books', 'Digital_Music', 'Movies_and_TV', 'Video_Games']
    # genre = ['Digital_Music']
    genre = ['Movies_and_TV']
    # genre = ['Books']
    # genre = ['AMAZON_FASHION', 'All_Beauty', 'Luxury_Beauty', 'Magazine_Subscriptions']

    def __init__(self, root, split, data_mode, target_mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.data_mode = data_mode
        self.target_mode = target_mode
        self.transform = transform
        if not check_exists(self.processed_folder):
            self.process()
        self.data, self.target = load(os.path.join(self.processed_folder, self.target_mode, '{}.pt'.format(self.split)),
                                      mode='pickle')
        if self.data_mode == 'user':
            pass
        elif self.data_mode == 'item':
            data_coo = self.data.tocoo()
            target_coo = self.target.tocoo()
            self.data = csr_matrix((data_coo.data, (data_coo.col, data_coo.row)),
                                   shape=(self.data.shape[1], self.data.shape[0]))
            self.target = csr_matrix((target_coo.data, (target_coo.col, target_coo.row)),
                                     shape=(self.target.shape[1], self.target.shape[0]))
        else:
            raise ValueError('Not valid data mode')
        # item_attr = load(os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        # self.item_attr = {'data': item_attr, 'target': item_attr}
        self.item_attr = {}

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
            if 'data' in self.item_attr:
                input['item_attr'] = torch.tensor(self.item_attr['data'][data.col])
            if 'target' in self.item_attr:
                input['target_item_attr'] = torch.tensor(self.item_attr['target'][target.col])
        elif self.data_mode == 'item':
            input = {'user': torch.tensor(data.col, dtype=torch.long),
                     'item': torch.tensor(np.array([index]), dtype=torch.long),
                     'rating': torch.tensor(data.data),
                     'target_user': torch.tensor(target.col, dtype=torch.long),
                     'target_item': torch.tensor(np.array([index]), dtype=torch.long),
                     'target_rating': torch.tensor(target.data)}
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
        train_set, test_set = self.make_explicit_data()
        save(train_set, os.path.join(self.processed_folder, 'explicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'explicit', 'test.pt'), mode='pickle')
        train_set, test_set = self.make_implicit_data()
        save(train_set, os.path.join(self.processed_folder, 'implicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'implicit', 'test.pt'), mode='pickle')
        # item_attr = self.make_info()
        # save(item_attr, os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        user = []
        item = []
        rating = []
        for i in range(len(self.genre)):
            data_i = pd.read_csv(os.path.join(self.raw_folder, '{}.csv'.format(self.genre[i])), header=None)
            user_i = data_i.iloc[:, 1].to_numpy()
            item_i = data_i.iloc[:, 0].to_numpy()
            item_id_i, item_inv_i = np.unique(item_i, return_inverse=True)
            item_id_map_i = {item_id_i[i]: i for i in range(len(item_id_i))}
            item_i = np.array([item_id_map_i[i] for i in item_id_i], dtype=np.int64)[item_inv_i].reshape(item_i.shape)
            rating_i = data_i.iloc[:, 2].astype(np.float32)
            user.append(user_i)
            if i > 0:
                item_i = item_i + len(item[i - 1])
            item.append(item_i)
            rating.append(rating_i)

        matched_user_id = []
        # for i in range(len(self.genre)):
        #     for j in range(i + 1, len(self.genre)):
        #         matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
        #         matched_user_id.append(matched_userid_i_j)
        
        for i in range(len(self.genre)):
            for j in range(1):
                matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
                matched_user_id.append(matched_userid_i_j)

        matched_user_id = np.array(list(set.intersection(*matched_user_id)))
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        rating = np.concatenate(rating, axis=0)

        user_id, user_inv = np.unique(user, return_inverse=True)
        _, _, unique_matched_user_inv = np.intersect1d(matched_user_id, user_id, return_indices=True)
        mask = np.isin(user_inv, unique_matched_user_inv)
        user = user[mask]
        item = item[mask]
        rating = rating[mask]

        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(user.shape)

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
        print('user, item, rating', user, item, rating, len(rating))
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        print('unique user / item', M, N)
        
        # sliced_user = user_id[:5000]
        # sliced_item = item_id[:9000]
        sliced_user = user_id[:6000]
        # sliced_item = item_id[:9000]
        sliced_item = item_id[:]
        print(f'sliced_user: {sliced_user}')
        sliced_mask = np.logical_and(np.isin(user, sliced_user), np.isin(item, sliced_item))

        user = user[sliced_mask]
        print(f'sliced_user: {user}')
        item = item[sliced_mask]
        rating = rating[sliced_mask]
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        print('sliced unique user / item', M, N, len(rating))
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)
        
        # Movies_and_TV
        # user: 20, item: 20
        # user, item, rating [2580305   82003 3646891 ... 2733966 1266990 2319410] [    18     18     18 ... 182014 182025 182025] [5. 5. 5. ... 5. 5. 5.] 1205169
        # unique user / item 28682 40742

        # user: 50, item: 200
        # user, item, rating [3806616    2233 2135155 ... 2135155 1569568 3260040] [    16     16     16 ... 181841 181841 182008] [5. 3. 3. ... 5. 5. 4.] 342294
        # unique user / item 5782 6922

        # user: 50, item: 300
        # user, item, rating [3806616    2233 2135155 ... 3372017  441555 3260040] [    16     16     16 ... 181488 181488 182008] [5. 3. 3. ... 5. 4. 4.] 288934
        # unique user / item 5762 4688    

        # books


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
        for i in range(len(self.genre)):
            data_i = pd.read_csv(os.path.join(self.raw_folder, '{}.csv'.format(self.genre[i])), header=None)
            user_i = data_i.iloc[:, 1].to_numpy()
            item_i = data_i.iloc[:, 0].to_numpy()
            item_id_i, item_inv_i = np.unique(item_i, return_inverse=True)
            item_id_map_i = {item_id_i[i]: i for i in range(len(item_id_i))}
            item_i = np.array([item_id_map_i[i] for i in item_id_i], dtype=np.int64)[item_inv_i].reshape(item_i.shape)
            rating_i = data_i.iloc[:, 2].astype(np.float32)
            user.append(user_i)
            if i > 0:
                item_i = item_i + len(item[i - 1])
            item.append(item_i)
            rating.append(rating_i)

        matched_user_id = []
        # for i in range(len(self.genre)):
        #     for j in range(i + 1, len(self.genre)):
        #         matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
        #         matched_user_id.append(matched_userid_i_j)
        
        for i in range(len(self.genre)):
            for j in range(1):
                matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
                matched_user_id.append(matched_userid_i_j)

        matched_user_id = np.array(list(set.intersection(*matched_user_id)))
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        rating = np.concatenate(rating, axis=0)

        user_id, user_inv = np.unique(user, return_inverse=True)
        _, _, unique_matched_user_inv = np.intersect1d(matched_user_id, user_id, return_indices=True)
        mask = np.isin(user_inv, unique_matched_user_inv)
        user = user[mask]
        item = item[mask]
        rating = rating[mask]

        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(user.shape)

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
        print('user, item, rating', user, item, rating, len(rating))
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        print('unique user / item', M, N)
        
        sliced_user = user_id[:6000]
        # sliced_item = item_id[:9000]
        sliced_item = item_id[:]
        print(f'sliced_user: {sliced_user}')
        sliced_mask = np.logical_and(np.isin(user, sliced_user), np.isin(item, sliced_item))

        user = user[sliced_mask]
        print(f'sliced_user: {user}')
        item = item[sliced_mask]
        rating = rating[sliced_mask]
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        print('sliced unique user / item', M, N, len(rating))
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)
        
        idx = np.random.permutation(user.shape[0])
        num_train = int(user.shape[0] * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        train_user, train_item, train_rating = user[train_idx], item[train_idx], rating[train_idx]
        train_rating[train_rating < 3.5] = 0
        train_rating[train_rating >= 3.5] = 1
        test_user, test_item, test_rating = user[test_idx], item[test_idx], rating[test_idx]
        test_rating[test_rating < 3.5] = 0
        test_rating[test_rating >= 3.5] = 1
        train_data = csr_matrix((train_rating, (train_user, train_item)), shape=(M, N))
        train_target = train_data
        test_data = train_data
        test_target = csr_matrix((test_rating, (test_user, test_item)), shape=(M, N))
        return (train_data, train_target), (test_data, test_target)

    def make_info(self):
        user = []
        item = []
        rating = []
        for i in range(len(self.genre)):
            data_i = pd.read_csv(os.path.join(self.raw_folder, '{}.csv'.format(self.genre[i])), header=None)
            user_i = data_i.iloc[:, 1].to_numpy()
            item_i = data_i.iloc[:, 0].to_numpy()
            item_id_i, item_inv_i = np.unique(item_i, return_inverse=True)
            item_id_map_i = {item_id_i[i]: i for i in range(len(item_id_i))}
            item_i = np.array([item_id_map_i[i] for i in item_id_i], dtype=np.int64)[item_inv_i].reshape(item_i.shape)
            rating_i = data_i.iloc[:, 2].astype(np.float32)
            user.append(user_i)
            if i > 0:
                item_i = item_i + len(item[i - 1])
            item.append(item_i)
            rating.append(rating_i)

        matched_user_id = []
        for i in range(len(self.genre)):
            for j in range(i + 1, len(self.genre)):
                matched_userid_i_j = set(user[i].tolist()).intersection(set(user[j].tolist()))
                matched_user_id.append(matched_userid_i_j)
        matched_user_id = np.array(list(set.intersection(*matched_user_id)))

        num_items_genre = [len(x) for x in item]
        user = np.concatenate(user, axis=0)
        item = np.concatenate(item, axis=0)
        rating = np.concatenate(rating, axis=0)

        user_id, user_inv = np.unique(user, return_inverse=True)
        _, _, unique_matched_user_inv = np.intersect1d(matched_user_id, user_id, return_indices=True)
        mask = np.in1d(user_inv, unique_matched_user_inv)
        user = user[mask]
        item = item[mask]
        rating = rating[mask]

        num_items_genre_ = []
        pivot = 0
        for i in range(len(num_items_genre)):
            num_items_i = int(mask[pivot:pivot + num_items_genre[i]].astype(np.float32).sum())
            pivot = pivot + num_items_genre[i]
            num_items_genre_.append(num_items_i)
        num_items_genre = num_items_genre_

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

        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)

        num_items_genre_ = []
        pivot = 0
        for i in range(len(num_items_genre)):
            num_items_i = int(dense_mask[pivot:pivot + num_items_genre[i]].astype(np.float32).sum())
            pivot = pivot + num_items_genre[i]
            num_items_genre_.append(num_items_i)
        num_items_genre = num_items_genre_
        item_id, item_inv = np.unique(item, return_inverse=True)
        item_attr = np.zeros((len(item_id), len(self.genre)), dtype=np.float32)
        pivot = 0
        for i in range(len(num_items_genre)):
            item_inv_i = np.intersect1d(item_id, item[pivot:pivot + num_items_genre[i]], return_indices=True)[1]
            item_attr[item_inv_i, i] += 1
            pivot = pivot + num_items_genre[i]
        return item_attr
