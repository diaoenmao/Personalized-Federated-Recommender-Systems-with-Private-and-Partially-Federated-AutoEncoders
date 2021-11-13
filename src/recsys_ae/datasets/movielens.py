import numpy as np
import scipy
import os
import torch
from torch.utils.data import Dataset
from utils import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file
from scipy.sparse import csr_matrix
from config import cfg


class ML100K(Dataset):
    data_name = 'ML100K'
    file = [('https://files.grouplens.org/datasets/movielens/ml-100k.zip', '0e33842e24a9c977be4e0107933c0723')]

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
        user_profile = load(os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        self.user_profile = {'data': user_profile, 'target': user_profile}
        item_attr = load(os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        self.item_attr = {'data': item_attr, 'target': item_attr}

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
        train_set, test_set = self.make_explicit_data()
        save(train_set, os.path.join(self.processed_folder, 'explicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'explicit', 'test.pt'), mode='pickle')
        train_set, test_set = self.make_implicit_data()
        save(train_set, os.path.join(self.processed_folder, 'implicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'implicit', 'test.pt'), mode='pickle')
        user_profile, item_attr = self.make_info()
        save(user_profile, os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        save(item_attr, os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-100k', 'u.data'), delimiter='\t')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-100k', 'u.data'), delimiter='\t')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        import pandas as pd
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        user_profile = pd.read_csv(os.path.join(self.raw_folder, 'ml-100k', 'u.user'), delimiter='|',
                                   names=['id', 'age', 'gender', 'occupation', 'zipcode'], encoding="latin",
                                   engine='python')
        age = user_profile['age'].to_numpy().astype(np.int64)
        age[age <= 17] = 0
        age[(age >= 18) & (age <= 24)] = 1
        age[(age >= 25) & (age <= 34)] = 2
        age[(age >= 35) & (age <= 44)] = 3
        age[(age >= 45) & (age <= 49)] = 4
        age[(age >= 50) & (age <= 55)] = 5
        age[age >= 56] = 6
        age = np.eye(7, dtype=np.float32)[age]
        gender = le.fit_transform(user_profile['gender'].to_numpy()).astype(np.int64)
        gender = np.eye(len(le.classes_), dtype=np.float32)[gender]
        occupation = le.fit_transform(user_profile['occupation'].to_numpy()).astype(np.int64)
        occupation = np.eye(len(le.classes_), dtype=np.float32)[occupation]
        user_profile = np.hstack([age, gender, occupation])
        item_attr = pd.read_csv(os.path.join(self.raw_folder, 'ml-100k', 'u.item'), delimiter='|', header=None,
                                encoding="latin", engine='python')
        genre = item_attr.iloc[:, 5:].to_numpy().astype(np.float32)
        item_attr = genre[:, 1:]
        return user_profile, item_attr


class ML1M(Dataset):
    data_name = 'ML1M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-1m.zip', 'c4d9eecfca2ab87c1945afe126590906')]

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
        user_profile = load(os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        self.user_profile = {'data': user_profile, 'target': user_profile}
        item_attr = load(os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        self.item_attr = {'data': item_attr, 'target': item_attr}

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
        train_set, test_set = self.make_explicit_data()
        save(train_set, os.path.join(self.processed_folder, 'explicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'explicit', 'test.pt'), mode='pickle')
        train_set, test_set = self.make_implicit_data()
        save(train_set, os.path.join(self.processed_folder, 'implicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'implicit', 'test.pt'), mode='pickle')
        user_profile, item_attr = self.make_info()
        save(user_profile, os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        save(item_attr, os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-1m', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-1m', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        import pandas as pd
        from sklearn import preprocessing
        le = preprocessing.LabelEncoder()
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-1m', 'ratings.dat'), delimiter='::')
        user, item, rating, ts = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(
            np.float32), data[:, 3].astype(np.float32)
        item_id, item_inv = np.unique(item, return_inverse=True)
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user_profile = pd.read_csv(os.path.join(self.raw_folder, 'ml-1m', 'users.dat'), delimiter='::',
                                   names=['id', 'gender', 'age', 'occupation', 'zipcode'], engine='python')
        age = le.fit_transform(user_profile['age'].to_numpy()).astype(np.int64)
        age = np.eye(len(le.classes_), dtype=np.float32)[age]
        gender = le.fit_transform(user_profile['gender'].to_numpy()).astype(np.int64)
        gender = np.eye(len(le.classes_), dtype=np.float32)[gender]
        occupation = le.fit_transform(user_profile['occupation'].to_numpy()).astype(np.int64)
        occupation = np.eye(len(le.classes_), dtype=np.float32)[occupation]
        user_profile = np.hstack([age, gender, occupation])
        item_attr = pd.read_csv(os.path.join(self.raw_folder, 'ml-1m', 'movies.dat'), delimiter='::',
                                names=['id', 'name', 'genre'], engine='python')
        item_attr = item_attr[item_attr['id'].isin(list(item_id_map.keys()))]
        genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                      'Western']
        genre_map = lambda x: [1 if g in x else 0 for g in genre_list]
        genre = np.array(item_attr['genre'].apply(genre_map).to_list(), dtype=np.float32)
        item_attr = genre
        return user_profile, item_attr


class ML10M(Dataset):
    data_name = 'ML10M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-10m.zip', 'ce571fd55effeba0271552578f2648bd')]

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
        item_attr = load(os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        self.item_attr = {'data': item_attr, 'target': item_attr}

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
        item_attr = self.make_info()
        save(item_attr, os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-10M100K', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-10M100K', 'ratings.dat'), delimiter='::')
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        import pandas as pd
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-10M100K', 'ratings.dat'), delimiter='::')
        user, item, rating, ts = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(
            np.float32), data[:, 3].astype(np.float32)
        item_id, item_inv = np.unique(item, return_inverse=True)
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        item_attr = pd.read_csv(os.path.join(self.raw_folder, 'ml-10M100K', 'movies.dat'), delimiter='::',
                                names=['id', 'name', 'genre'], engine='python')
        item_attr = item_attr[item_attr['id'].isin(list(item_id_map.keys()))]
        genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                      'Western']
        genre_map = lambda x: [1 if g in x else 0 for g in genre_list]
        genre = np.array(item_attr['genre'].apply(genre_map).to_list(), dtype=np.float32)
        item_attr = genre
        return item_attr


class ML20M(Dataset):
    data_name = 'ML20M'
    file = [('https://files.grouplens.org/datasets/movielens/ml-20m.zip', 'cd245b17a1ae2cc31bb14903e1204af3')]

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
        item_attr = load(os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        self.item_attr = {'data': item_attr, 'target': item_attr}

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
        item_attr = self.make_info()
        save(item_attr, os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, md5) in self.file:
            filename = os.path.basename(url)
            download_url(url, self.raw_folder, filename, md5)
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split)
        return fmt_str

    def make_explicit_data(self):
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-20m', 'ratings.csv'), delimiter=',', skip_header=1)
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-20m', 'ratings.csv'), delimiter=',', skip_header=1)
        user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
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
        import pandas as pd
        data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-20m', 'ratings.csv'), delimiter=',', skip_header=1)
        user, item, rating, ts = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(
            np.float32), data[:, 3].astype(np.float32)
        item_id, item_inv = np.unique(item, return_inverse=True)
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        item_attr = pd.read_csv(os.path.join(self.raw_folder, 'ml-20m', 'movies.csv'), delimiter=',')
        item_attr = item_attr[item_attr['movieId'].isin(list(item_id_map.keys()))]
        genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                      'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                      'Western']
        genre_map = lambda x: [1 if g in x else 0 for g in genre_list]
        genre = np.array(item_attr['genres'].apply(genre_map).to_list(), dtype=np.float32)
        genome_tags = pd.read_csv(os.path.join(self.raw_folder, 'ml-20m', 'genome-tags.csv'), delimiter=',',
                                  dtype={'tagId': int, 'tag': str})
        item_attr = pd.read_csv(os.path.join(self.raw_folder, 'ml-20m', 'genome-scores.csv'), delimiter=',',
                                dtype={'movieId': int, 'tagId': int, 'relevance': float})
        item = item_attr['movieId'].astype(np.int64)
        valid_mask = np.in1d(item, item_id)
        item_attr = item_attr[valid_mask]
        relevance = np.zeros((len(item_id), len(genome_tags['tagId'])), dtype=np.float32)
        item = item_attr['movieId'].astype(np.int64)
        item_id, item_inv = np.unique(item, return_inverse=True)
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)
        tag_id = item_attr['tagId'].to_numpy().astype(np.int64)
        relevance[item, tag_id - 1] = item_attr['relevance'].to_numpy()
        item_attr = np.hstack([genre, relevance])
        return item_attr
