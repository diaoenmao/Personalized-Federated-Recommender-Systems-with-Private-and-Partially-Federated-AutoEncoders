import copy
from platform import node
import torch
import models
import collections
import numpy as np
import torch.nn as nn

from config import cfg
from utils import make_optimizer, make_scheduler, save, load
from collections import OrderedDict

class Federation:
    def __init__(self, data_split_info):

        self.data_split_info = data_split_info
        
        self.local_model_dict = collections.defaultdict(dict)
        self.local_optimizer_state_dict = {}

        if cfg['experiment_size'] == 'large':
            self.server_model = eval('models.{}().to("cpu")'.format(cfg['model_name']))
        else:
            self.server_model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))

        self.server_optimizer = make_optimizer(self.server_model, cfg['model_name'], 'server')
        self.server_optimizer_state_dict = self.server_optimizer.state_dict()
        
        self.new_server_model_parameter_dict = collections.defaultdict(int)
        self.items_for_each_user = collections.defaultdict(set)

        self.item_union_set_pre_calculated = collections.defaultdict(int)
        

    def reset_parameters(self, model):
        for model in [model.encoder, model.decoder]:
            for m in model.blocks:
                # if item m is nn.Linear, set its value to xavier_uniform heuristic value
                if isinstance(m, nn.Linear):
                # if not isinstance(m, nn.Tanh):
                    m.weight.data.zero_()
                    if m.bias is not None:
                        # set bias to 0
                        m.bias.data.zero_()

    def record_items_for_each_user(self, dataset):
        '''
        Record each user's items
        '''
        for i in range(cfg['num_users']['data']):
            for item in list(dataset[i]['item']):
                self.items_for_each_user[i].add(item.item())
        return True

    def calculate_item_union_set(self, node_idx, user_list):
        '''
        For each model at each round, calculate how many items(points in the last layer)
        have been activated
        '''
        if node_idx in self.item_union_set_pre_calculated:
            return self.item_union_set_pre_calculated[node_idx]

        item_union_set = set()
        for user in user_list:
            item_union_set = item_union_set | self.items_for_each_user[user]
        self.item_union_set_pre_calculated[node_idx] = item_union_set
        return self.item_union_set_pre_calculated[node_idx]


    def store_local_model(self, index, model):
        self.local_model_dict[index] = copy.deepcopy(model)
        return

    def load_local_model(self, index):
        return self.local_model_dict[index]
    
    def store_local_optimizer_state_dict(self, cur_node_index, local_optimizer_state_dict):
        self.local_optimizer_state_dict[cur_node_index] = copy.deepcopy(local_optimizer_state_dict)
        return 

    def get_local_optimizer_state_dict(self, cur_node_index):
        return self.local_optimizer_state_dict[cur_node_index]

    def create_local_model_and_local_optimizer(self):
        
        for i in range(cfg['num_nodes']):
            
            local_model = copy.deepcopy(self.server_model)
            local_optimizer = make_optimizer(local_model, cfg['model_name'], 'client')
            if cfg['info_size'] is not None:
                if 'user_profile' in cfg['info_size']:
                    local_model.user_profile.load_state_dict(copy.deepcopy(self.server_model.user_profile.state_dict()))

            self.store_local_model(i, local_model)
            self.store_local_optimizer_state_dict(i, local_optimizer.state_dict())

            if i % int((cfg['num_nodes'] * cfg['log_interval']) + 1) == 0:
                print('create_local_model', i)
        
        return
    
    def create_local_optimizer(self):
        
        for i in range(cfg['num_nodes']):
            
            local_model = copy.deepcopy(self.server_model)
            local_optimizer = make_optimizer(local_model, cfg['model_name'], 'client')
            self.store_local_optimizer_state_dict(i, local_optimizer.state_dict())

            if i % int((cfg['num_nodes'] * cfg['log_interval']) + 1) == 0:
                print('create_local_model', i)
        
        return

    def update_client_parameters_with_server_model_parameters(self, cur_model):
        if cfg['federated_mode'] == 'decoder':       
            net_model_dict = {}
            # for key in self.server_model.state_dict():
            for key,value in self.server_model.named_parameters():
                if cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                else:   
                    net_model_dict[key] = copy.deepcopy(self.server_model.state_dict()[key])
            cur_model.load_state_dict(copy.deepcopy(net_model_dict), strict=False)
        elif cfg['federated_mode'] == 'all':
            cur_model.load_state_dict(copy.deepcopy(self.server_model.state_dict()))  
        else:
            raise ValueError('Not valid federated mode')
        return cur_model

    def generate_new_server_model_parameter_dict(self, model_state_dict, total_client, item_union_set=None):

        if cfg['model_name'] != 'ae':
            raise ValueError('model_name is not ae')
        if cfg['train_mode'] != 'fedavg':
            raise ValueError('train_mode is not fedavg')

        if cfg['federated_mode'] == 'decoder':
            for key, value in self.server_model.named_parameters():
                if cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                    # dont calculate encoder for server_model because encoder is unique for each client
                    # when cfg['federated_mode'] == 'decoder'
                    pass
                else:
                    cur_ratio = 1 / total_client
                    self.new_server_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(model_state_dict[key]))
        elif cfg['federated_mode'] == 'all':
            for key in model_state_dict:
                cur_ratio = 1 / total_client
                self.new_server_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(model_state_dict[key]))
        else:   
            raise ValueError('Not valid model name')
        return

    def update_server_model_momentum(self):
        with torch.no_grad():
            server_optimizer = make_optimizer(self.server_model, cfg['model_name'], 'server')
            server_optimizer.load_state_dict(copy.deepcopy(self.server_optimizer_state_dict))
            server_optimizer.zero_grad()
            c = next(self.server_model.parameters()).device
            for k,v in self.server_model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = copy.deepcopy(v.data.new_zeros(v.size()))
                    tmp_v += copy.deepcopy(self.new_server_model_parameter_dict[k])
                    v.grad = (v.data - tmp_v).detach()

            server_optimizer.step()
            self.server_optimizer_state_dict = server_optimizer.state_dict()
            self.new_server_model_parameter_dict = collections.defaultdict(int)
        return
