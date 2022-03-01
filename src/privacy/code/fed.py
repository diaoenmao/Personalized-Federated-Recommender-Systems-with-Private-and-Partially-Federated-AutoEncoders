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

        cur_num_users = self.data_split_info[0]['num_users']
        cur_num_items = self.data_split_info[0]['num_items']
        self.global_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
        self.global_optimizer = make_optimizer(self.global_model, cfg['model_name'])
        self.global_optimizer_state_dict = self.global_optimizer.state_dict()
        
        self.new_global_model_parameter_dict = collections.defaultdict(int)
        self.global_grade_item_for_user = collections.defaultdict(set)

        # self.batch_normalization_name = {}
        # self.local_scheduler = {}
        # self.partial_average_state_dict = copy.deepcopy(self.global_model.state_dict())
        # self.batch_normalization_name = {'encoder.blocks.1.weight', 'encoder.blocks.1.bias', 'encoder.blocks.4.weight', 'encoder.blocks.4.bias'}

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

    def record_global_grade_item_for_user(self, dataset):
        for i in range(cfg['num_users']['data']):
            for item in list(dataset[i]['item']):
                self.global_grade_item_for_user[i].add(item.item())
        return True

    def calculate_item_iteraction_set(self, user_list):
        item_iteraction_set = set()
        for user in user_list:
            item_iteraction_set = item_iteraction_set | self.global_grade_item_for_user[user]
        return item_iteraction_set

    def distribute(self, model):
        model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))  
        return model

    def get_partial_average_state_dict(self):
        temp_partial_average_state_dict = copy.deepcopy(self.partial_average_state_dict)
        self.partial_average_state_dict = collections.defaultdict(int)
        return temp_partial_average_state_dict

    def store_partial_average_state_dict(self, cur_model_train_state_dict, num_average_nodes):
        for key in cur_model_train_state_dict:
            # print(num_average_nodes, value, type(value))
            self.partial_average_state_dict[key] += (1/num_average_nodes) * copy.deepcopy(cur_model_train_state_dict[key]) 
        return

    def store_local_model(self, index, model):
        # if cfg['control']['num_nodes'] == 'max' and cfg['federated_mode'] == 'decoder':
        #     # print('index', index)
        #     save(model, '../decoder/model/{}/{}.pt'.format(cfg['model_tag'], index))
        # else:
        self.local_model_dict[index] = copy.deepcopy(model)
        return True

    def load_local_model(self, index):
        # if cfg['control']['num_nodes'] == 'max' and cfg['federated_mode'] == 'decoder':
        #     return load('../decoder/model/{}/{}.pt'.format(cfg['model_tag'], index))
        # else:
        return self.local_model_dict[index]
    
    def store_local_optimizer_state_dict(self, cur_node_index, local_optimizer_state_dict):
        # if cfg['control']['num_nodes'] == 'max' and cfg['federated_mode'] == 'decoder':
        #     save(local_optimizer, '../decoder/local_optimizer/{}/{}.pt'.format(cfg['model_tag'], cur_node_index))
        # else:
        self.local_optimizer_state_dict[cur_node_index] = copy.deepcopy(local_optimizer_state_dict)
        return 

    def get_local_optimizer_state_dict(self, cur_node_index):
        # if cfg['control']['num_nodes'] == 'max' and cfg['federated_mode'] == 'decoder':
        #     return load('../decoder/local_optimizer/{}/{}.pt'.format(cfg['model_tag'], cur_node_index))
        # else:

        return self.local_optimizer_state_dict[cur_node_index]

    def create_local_model_and_local_optimizer(self):
        
        for i in range(cfg['num_nodes']):
            
            local_model = copy.deepcopy(self.global_model)
            local_optimizer = make_optimizer(local_model, cfg['model_name'])
            if cfg['info_size'] is not None:
                if 'user_profile' in cfg['info_size']:
                    local_model.user_profile.load_state_dict(copy.deepcopy(self.global_model.user_profile.state_dict()))
                if 'item_attr' in cfg['info_size']:
                    local_model.item_attr.load_state_dict(copy.deepcopy(self.global_model.item_attr.state_dict()))

            self.store_local_model(i, local_model)
            self.store_local_optimizer_state_dict(i, local_optimizer.state_dict())

            if i % int((cfg['num_nodes'] * cfg['log_interval']) + 1) == 0:
                print('create_local_model', i)
        
        return

    def update_client_parameters_with_global_parameters(self, cur_model):
        
        if cfg['federated_mode'] == 'decoder':       
            net_model_dict = {}
            # for key in self.global_model.state_dict():
            for key,value in self.global_model.named_parameters():
                if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key]) 
                elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                else:   
                    net_model_dict[key] = copy.deepcopy(self.global_model.state_dict()[key])
            cur_model.load_state_dict(copy.deepcopy(net_model_dict), strict=False)
        else:
            raise ValueError('Not valid federated mode')
        return cur_model

    def generate_new_global_model_parameter_dict(self, model_state_dict, total_client, item_iteraction_set=None):
        # print('total_clinet', total_client)
        if cfg['model_name'] == 'ae':
            if cfg['train_mode'] == 'fedavg':
                if cfg['federated_mode'] == 'decoder':
                    for key, value in self.global_model.named_parameters():
                        if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                            pass
                        elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                            pass
                        elif (key == 'decoder.blocks.3.weight' or key == 'decoder.blocks.3.bias') and item_iteraction_set:
                            cur_ratio = 1 / total_client
                            global_state_dict = self.global_model.state_dict()

                            if key not in self.new_global_model_parameter_dict:
                                self.new_global_model_parameter_dict[key] = copy.deepcopy(global_state_dict[key].new_zeros(global_state_dict[key].size()))

                            for i in range(self.new_global_model_parameter_dict[key].size()[0]):
                                if i in item_iteraction_set:
                                    self.new_global_model_parameter_dict[key][i] += (cur_ratio * copy.deepcopy(model_state_dict[key][i]))
                                else:
                                    self.new_global_model_parameter_dict[key][i] += (cur_ratio * copy.deepcopy(global_state_dict[key][i]))
                        else:
                            cur_ratio = 1 / total_client
                            self.new_global_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(model_state_dict[key]))
                if cfg['federated_mode'] == 'all':
                    for key in model_state_dict:
                        cur_ratio = 1 / total_client
                        self.new_global_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(model_state_dict[key]))
        else:   
            raise ValueError('Not valid model name')
        return

    def update_global_model_momentum(self):
        with torch.no_grad():
            global_optimizer = make_optimizer(self.global_model, cfg['model_name'])
            global_optimizer.load_state_dict(copy.deepcopy(self.global_optimizer_state_dict))
            global_optimizer.zero_grad()
            for k,v in self.global_model.named_parameters():
                parameter_type = k.split('.')[-1]
                if 'weight' in parameter_type or 'bias' in parameter_type:
                    tmp_v = copy.deepcopy(v.data.new_zeros(v.size()))
                    tmp_v += copy.deepcopy(self.new_global_model_parameter_dict[k])
                      # a = v.data
                    #requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
                    #v.data(t-1) - 1 * (v.data(t-1) - tmp_v(t)) = tmp_v(t)
                    v.grad = (v.data - tmp_v).detach()

            global_optimizer.step()
            self.global_optimizer_state_dict = global_optimizer.state_dict()
            self.new_global_model_parameter_dict = collections.defaultdict(int)
        return


