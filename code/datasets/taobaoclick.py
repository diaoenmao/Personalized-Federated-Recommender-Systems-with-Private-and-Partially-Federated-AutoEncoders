# from curses import raw
import numpy as np
import scipy
import os
import math
import torch
import random
import pandas as pd
from torch.utils.data import Dataset

from .datasets_utils import download_url, extract_file, if_value_is_nan, change_to_absolute_path
from scipy.sparse import csr_matrix
# from config import cfg

# import sys
# sys.path.append("../utils")
# from utils import 
from utils import check_exists, makedir_exist_ok, save, load

class taobaoclicksmall(Dataset):
    
    data_name = 'taobaoclick'
    file = [('https://files.grouplens.org/datasets/movielens/ml-1m.zip', 'c4d9eecfca2ab87c1945afe126590906')]

    def __init__(self, root, split, data_mode, target_mode, transform=None):
        self.root = os.path.expanduser(root)
        self.split = split
        self.data_mode = data_mode
        self.target_mode = target_mode
        self.transform = transform
        self.selected_user_id_num = 6000
        self.selected_ad_id_num = 3000
        self.process()
        # if not check_exists(self.processed_folder):
        #     self.process()
        # self.data, self.target = load(os.path.join(self.processed_folder, self.target_mode, '{}.pt'.format(self.split)),
        #                               mode='pickle')
        # if self.data_mode == 'user':
        #     pass
        # elif self.data_mode == 'item':
        #     data_coo = self.data.tocoo()
        #     target_coo = self.target.tocoo()
        #     self.data = csr_matrix((data_coo.data, (data_coo.col, data_coo.row)),
        #                            shape=(self.data.shape[1], self.data.shape[0]))
        #     self.target = csr_matrix((target_coo.data, (target_coo.col, target_coo.row)),
        #                            shape=(self.target.shape[1], self.target.shape[0]))
        # else:
        #     raise ValueError('Not valid data mode')
        # # user_profile = load(os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        # # self.user_profile = {'data': user_profile, 'target': user_profile}
        # # item_attr = load(os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        # # self.item_attr = {'data': item_attr, 'target': item_attr}
        # self.user_profile = {}
        # self.item_attr = {}

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
        # return os.path.join(self.root, 'processed')
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        
        return os.path.join(self.root, 'raw')

    @property
    def rough_process_folder(self):
        return os.path.join(self.root, 'rough_process')

    def select_user_id(self):
        
        print('Start selecting user id')
        user_profile_data = pd.read_csv(change_to_absolute_path(os.path.join(self.rough_process_folder, 'rough_process_user_profile.csv')), delimiter=',',
                                   names=['user_id', 'cms_segid', 'cms_group_id', 'gender', 'age_level', 'consumption_level', 
                                          'shopping_level', 'is_college_student', 'city_level'], encoding="latin", engine='python')

        # user_profile_data['user_id'].iloc(0)
        user_id = user_profile_data['user_id'].to_numpy()
        if self.selected_user_id_num < 1 or self.selected_user_id_num > len(user_id):
            raise ValueError('Please input correct selected_user_id_num')
        user_id_index = [i for i in range(len(user_id))]  
        selected_user_id_index = random.sample(user_id_index, self.selected_user_id_num)
        selected_user_id = user_id[selected_user_id_index]

        user_profile_data = user_profile_data[user_profile_data.user_id.isin(selected_user_id)]
        return user_profile_data
    
    def select_ad_id(self):

        print('Start selecting ad id')
        ad_feature_data = pd.read_csv(change_to_absolute_path(os.path.join(self.rough_process_folder, 'rough_process_ad_feature.csv')), delimiter=',',
                                   names=['ad_id', 'category_id', 'campaign_id', 'customer_id', 'brand', 'price'], encoding="latin", engine='python')
        
        ad_id = ad_feature_data['ad_id'].to_numpy()
        if self.selected_ad_id_num < 1 or self.selected_ad_id_num > len(ad_id):
            raise ValueError('Please input correct selected_ad_id_num')
        
        ad_id_index = [i for i in range(len(ad_id))]  
        selected_ad_id_index = random.sample(ad_id_index, self.selected_ad_id_num)
        selected_ad_id = ad_id[selected_ad_id_index]

        ad_feature_data = ad_feature_data[ad_feature_data.ad_id.isin(selected_ad_id)]
        return ad_feature_data

    def select_raw_sample(self, user_profile_data, ad_feature_data):
        
        print('Start selecting raw sample')
        selected_user_id = user_profile_data['user_id'].to_numpy()
        selected_ad_id = ad_feature_data['ad_id'].to_numpy()

        raw_sample_data = pd.read_csv(change_to_absolute_path(os.path.join(self.rough_process_folder, 'rough_process_raw_sample.csv')), delimiter=',',
                                   names=['user_id', 'time_stamp', 'ad_id', 'pid', 'noclk', 'clk'], encoding="latin", engine='python')

        raw_data_based_on_user_id = raw_sample_data[raw_sample_data.user_id.isin(selected_user_id)]
        raw_data_based_on_user_id = raw_data_based_on_user_id.sort_values('clk', axis=0)
        # if user made judgement on a ad many times, we take the last one
        # This will cover the situation the user clicks the ad since the raw_sample_data is sorted
        raw_data_based_on_user_id = raw_data_based_on_user_id.drop_duplicates(['user_id', 'ad_id'], keep='last')
        ad_id = raw_data_based_on_user_id['ad_id'].to_numpy()
        unique_item_count = len(set(ad_id))
        print('unique_item_count', unique_item_count)
        print('row_count', len( raw_data_based_on_user_id['user_id'].to_numpy()))

        raw_data_based_on_ad_id = raw_sample_data[raw_sample_data.ad_id.isin(selected_ad_id)]
        user_id = raw_data_based_on_ad_id['user_id'].to_numpy()
        unique_user_count = len(set(user_id))
        print('unique_user_count', unique_user_count)

        raw_sample_data = raw_sample_data[raw_sample_data.user_id.isin(selected_user_id) & raw_sample_data.ad_id.isin(selected_ad_id)]
        raw_sample_data = raw_sample_data.sort_values('clk', axis=0)
        # if user made judgement on a ad many times, we take the last one
        # This will cover the situation the user clicks the ad since the raw_sample_data is sorted
        raw_sample_data = raw_sample_data.drop_duplicates(['user_id', 'ad_id'], keep='last')
        print('interaction', len(raw_sample_data['user_id']))
        return raw_sample_data

    # def ceshi(self):
    #     ad_feature_data = pd.read_csv(change_to_absolute_path(os.path.join(self.rough_process_folder, 'rough_process_ad_feature.csv')), delimiter=',',
    #                             names=['ad_id', 'category_id', 'campaign_id', 'customer_id', 'brand', 'price'], encoding="latin", engine='python', nrows=10)
        
    #     temp1 = ad_feature_data[ad_feature_data.ad_id.isin([63133])]
    #     temp2 = ad_feature_data[ad_feature_data.ad_id.isin(['63133'])]

    #     return

    def process(self):
        a = self.raw_folder
        res = check_exists(self.raw_folder)
        if not check_exists(self.raw_folder):
            taobaoclick_data_website = 'https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom'
            raise ValueError('download data from: ' + taobaoclick_data_website)
            # self.download()
        
        # Rough process the taobaoclick.csv
        if not check_exists(self.rough_process_folder):
            self.rough_process()
            self.rank_rough_data_by_user_(data_point_count=100000)

        # self.ceshi()
       
        train_set, test_set = self.make_implicit_data(user_profile_data, ad_feature_data, raw_sample_data)
        save(train_set, os.path.join(self.processed_folder, 'implicit', 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'implicit', 'test.pt'), mode='pickle')
        user_profile, item_attr = self.make_info()
        save(user_profile, os.path.join(self.processed_folder, 'user_profile.pt'), mode='pickle')
        save(item_attr, os.path.join(self.processed_folder, 'item_attr.pt'), mode='pickle')
        return

    def rough_process(self):
        picked_user_id = self.rough_process_user_profile()
        self.rough_process_raw_sample(picked_user_id)
        self.rough_process_ad_feature()

    def rough_process_user_profile(self):
        '''
            userid: 脱敏过的用户ID
            cms_segid: 微群ID
            cms_group_id: cms_group_id
            gender: 性别 1:男,2:女；
            age_level: 年龄层次
            consumption_level: 1: 低档, 2: 中档, 3: 高档
            shopping_level: 购物深度, 1:浅层用户,2:中度用户,3:深度用户
            is_college_student: 1:是,0:否
            city_level: 城市层级
        '''
        print('Start rough process user_profile.csv')
        data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'user_profile.csv')), delimiter=',',
                                   names=['user_id', 'cms_segid', 'cms_group_id', 'gender', 'age_level', 'consumption_level', 
                                          'shopping_level', 'is_college_student', 'city_level'], encoding="latin", engine='python', skiprows=1)

        # data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'user_profile.csv')), delimiter=',',
        #                         names=['user_id'], encoding="latin", engine='python', skiprows=1, usecols=[0])

        # data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'user_profile.csv')), 
        #                            encoding="latin", engine='python', skiprows=1)

        # for i in range(10):
        # print(data.iloc[0])
        print('total_user_count: ', len(data['user_id']))

        # delete rows that contain missing value
        data = data.dropna(axis=0)

       
        # user_id, cms_segid, cms_group_id, gender, age_level, consumption_level, shopping_level, is_college_student, city_level = (
        #     data['user_id'].to_numpy().astype(np.int64), 
        #     data['cms_segid'].to_numpy().astype(np.int64), 
        #     data['cms_group_id'].to_numpy().astype(np.int64),
        #     data['gender'].to_numpy().astype(np.int64),
        #     data['age_level'].to_numpy().astype(np.int64),
        #     data['consumption_level'].to_numpy().astype(np.int64),
        #     data['shopping_level'].to_numpy().astype(np.int64),
        #     data['is_college_student'].to_numpy().astype(np.int64),
        #     data['city_level'].to_numpy().astype(np.int64))

        user_id = data['user_id'].to_numpy().astype(np.int64)
        unique_user_id_count = len(user_id)
        print('unique_user_id_count: ', unique_user_id_count)
        print('user_containing_all_information_count: ', unique_user_id_count)

        data = pd.DataFrame(data)
        makedir_exist_ok(change_to_absolute_path(self.rough_process_folder))
        data.to_csv(change_to_absolute_path(os.path.join(self.rough_process_folder, 'rough_process_user_profile.csv')), encoding='latin', index=False)

        print('Rough process user_profile.csv done')

        picked_user_id = user_id
        return picked_user_id

    def rough_process_raw_sample(self, picked_user_id):
        '''
        user_id: 脱敏过的用户ID;
        adgroup_id: 脱敏过的广告单元ID;
        time_stamp: 时间戳；
        pid: 资源位；
        noclk: 为1代表没有点击, 为0代表点击;
        clk: 为0代表没有点击, 为1代表点击;
        '''
        print('Start rough process raw_sample.csv')
        # data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'raw_sample.csv')), delimiter=',',
        #                            names=['user_id', 'ad_id', 'time_stamp', 'pid', 'noclk', 'clk'], encoding="latin", engine='python', skiprows=1)

        # data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'raw_sample.csv')), delimiter=',',
        #                            names=['user_id'], encoding="latin", engine='python', skiprows=1, usecols=[0])

        data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'raw_sample.csv')), delimiter=',',
                                   names=['user_id', 'time_stamp', 'ad_id', 'pid', 'noclk', 'clk'], encoding="latin", engine='python', skiprows=1)

        print('raw_sample_total_row', len(data['user_id']))
        print('sadfasdfassass')
        # delete rows that contain missing value
        # data = data.dropna(axis=0)
        # user_id = data['user_id'].to_numpy().astype(np.int64)
        # user_id, ad_id, time_stamp, pid, noclk, clk = (
        #     data['user_id'].to_numpy().astype(np.int64), 
        #     data['ad_id'].to_numpy().astype(np.int64), 
        #     data['time_stamp'].to_numpy().astype(np.int64),
        #     data['pid'].to_numpy().astype(np.int64),
        #     data['noclk'].to_numpy().astype(np.int64),
        #     data['clk'].to_numpy().astype(np.int64))

        data = data[data.user_id.isin(picked_user_id)]


        # data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'raw_sample.csv')), delimiter=',',
        #                            names=['user_id', 'ad_id', 'time_stamp', 'pid', 'noclk', 'clk'], encoding="latin", engine='python', skiprows=skiprows)
        print('rough_process_raw_sample_total_row', len(data))
        data = pd.DataFrame(data)
        print('iiiiii')
        data.to_csv(change_to_absolute_path(os.path.join(self.rough_process_folder, 'rough_process_raw_sample.csv')), encoding='latin', index=False)

        print('Rough process raw_sample.csv done')
        return

    def rough_process_ad_feature(self):
        '''
            ad_id: 脱敏过的广告ID
            category_id: 脱敏过的商品类目ID
            campaign_id: 脱敏过的广告计划ID
            customer_id: 脱敏过的广告主ID
            brand: 脱敏过的品牌ID
            price: 宝贝的价格
        '''
        print('Start Rough process ad_feature.csv')
        data = pd.read_csv(change_to_absolute_path(os.path.join(self.raw_folder, 'ad_feature.csv')), delimiter=',',
                                   names=['ad_id', 'category_id', 'campaign_id', 'customer_id', 'brand', 'price'],
                                   encoding="latin", engine='python', skiprows=1)

        data = pd.DataFrame(data)
        print('iiiiii')
        data.to_csv(change_to_absolute_path(os.path.join(self.rough_process_folder, 'rough_process_ad_feature.csv')), encoding='latin', index=False)

        print('Rough process ad_feature.csv done')
        # delete rows that contain missing value
        # data = data.dropna(axis=0)


        # ad_id corresponds to a commodity
        # a commodity belongs to a ad_category_id
        # a commodity belongs to a brand_id
        # ad_id, category_id, campaign_id, customer_id, brand, price = (
        #     data['ad_id'].astype(np.int64), 
        #     data['category_id'].astype(np.int64), 
        #     data['campaign_id'].astype(np.int64),
        #     data['customer_id'].astype(np.int64),
        #     data['brand'].astype(np.int64),
        #     data['price'].astype(np.int64))

        return

    def rank_rough_data_by_user_(self, data_point_count):
        user_profile_data = self.select_user_id()
        ad_feature_data = self.select_ad_id()
        raw_sample_data = self.select_raw_sample(user_profile_data, ad_feature_data)





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

    def make_implicit_data(self, user_profile_data, ad_feature_data, raw_sample_data):
        # data = np.genfromtxt(os.path.join(self.raw_folder, 'ml-1m', 'ratings.dat'), delimiter='::')
        # user, item, rating = data[:, 0].astype(np.int64), data[:, 1].astype(np.int64), data[:, 2].astype(np.float32)
        # user_id, user_inv = np.unique(user, return_inverse=True)
        # item_id, item_inv = np.unique(item, return_inverse=True)
        # M, N = len(user_id), len(item_id)
        # user_id_map = {user_id[i]: i for i in range(len(user_id))}
        # item_id_map = {item_id[i]: i for i in range(len(item_id))}
        # user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
        # item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)

        # sort the raw_sample_data by clk value
        user = raw_sample_data['user_id'].to_numpy()        
        item = raw_sample_data['ad_id'].to_numpy()
        rating = raw_sample_data['clk'].to_numpy().astype(np.int64)

        a1,b1,c1 = len(user), len(item), len(rating)

        raw_sample_data = raw_sample_data.sort_values('clk', axis=0)
        # if user made judgement on a ad many times, we take the last one
        # This will cover the situation the user clicks the ad since the raw_sample_data is sorted
        raw_sample_data = raw_sample_data.drop_duplicates(['user_id', 'ad_id'], keep='last')

        user = raw_sample_data['user_id'].to_numpy()        
        item = raw_sample_data['ad_id'].to_numpy()
        rating = raw_sample_data['clk'].to_numpy().astype(np.int64)

        a,b,c = len(user), len(item), len(rating)
        user_id, user_inv = np.unique(user, return_inverse=True)
        item_id, item_inv = np.unique(item, return_inverse=True)
        M, N = len(user_id), len(item_id)
        user_id_map = {user_id[i]: i for i in range(len(user_id))}
        item_id_map = {item_id[i]: i for i in range(len(item_id))}
        user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(len(user))
        item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(len(item))

        idx = np.random.permutation(N)
        num_train = int(N * 0.9)
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        
        train_user, train_item, train_rating = user[train_idx], item[train_idx], rating[train_idx]
        # train_rating[train_rating < 3.5] = 0
        # train_rating[train_rating >= 3.5] = 1
        test_user, test_item, test_rating = user[test_idx], item[test_idx], rating[test_idx]
        # test_rating[test_rating < 3.5] = 0
        # test_rating[test_rating >= 3.5] = 1
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

