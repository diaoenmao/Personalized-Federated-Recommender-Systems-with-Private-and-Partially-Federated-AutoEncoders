import copy
from platform import node
import torch
import models
import collections
import numpy as np
import torch.nn as nn

from config import cfg
from utils import make_optimizer, make_scheduler
from collections import OrderedDict

class Federation:
    def __init__(self, data_split_info):
        # 清零
        self.data_split_info = data_split_info
        
        self.local_model_dict = collections.defaultdict(dict)
        self.local_test_model_dict = collections.defaultdict(dict)

        cur_num_users = self.data_split_info[0]['num_users']
        cur_num_items = self.data_split_info[0]['num_items']
        self.global_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
        global_optimizer = make_optimizer(self.global_model, 'global')
        self.global_optimizer_state_dict = global_optimizer.state_dict()

        self.new_global_model_parameter_dict = collections.defaultdict(int)
        self.batch_normalization_name = {}
        self.local_optimizer = {}
        self.local_scheduler = {}
        # self.batch_normalization_name = {'encoder.blocks.1.weight', 'encoder.blocks.1.bias', 'encoder.blocks.4.weight', 'encoder.blocks.4.bias'}

    def get_local_optimizer(self, cur_node_index, model):
        if cur_node_index in self.local_optimizer:
            return self.local_optimizer[cur_node_index]
        
        optimizer = make_optimizer(model, cfg['model_name'])
        self.local_optimizer[cur_node_index] = optimizer
        return self.local_optimizer[cur_node_index]

    def get_local_scheduler(self, cur_node_index, model):
        if cur_node_index in self.local_scheduler:
            return self.local_scheduler[cur_node_index]
        
        scheduler = make_scheduler(self.get_local_optimizer(cur_node_index, model), cfg['model_name'])
        self.local_scheduler[cur_node_index] = scheduler
        return self.local_scheduler[cur_node_index]

    def load_local_model_dict(self, index):
        return self.local_model_dict[index]

    def load_local_test_model_dict(self, index):
        return self.local_test_model_dict[index]

    def get_new_global_model_parameter_dict(self):
        return self.new_global_model_parameter_dict

    def update_global_model_parameters(self):
        self.global_model.load_state_dict(copy.deepcopy(self.new_global_model_parameter_dict))        
        return

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

    # def distribute(self, model, momentum_parameter=False):
    #     if momentum_parameter == True:
    #         model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
    #     else:
    #         model.load_state_dict(copy.deepcopy(self.new_global_model_parameter_dict))
    #     return model

    def distribute(self, model):
        model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))  
        return model

    def create_local_model_dict(self):

        for i in range(cfg['num_nodes']):

            cur_num_users = self.data_split_info[i]['num_users']
            cur_num_items = self.data_split_info[i]['num_items']
            local_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
            
            test_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))

            local_model.encoder.load_state_dict(copy.deepcopy(self.global_model.encoder.state_dict()))
            local_model.decoder.load_state_dict(copy.deepcopy(self.global_model.decoder.state_dict()))
                
            if cfg['info_size'] is not None:
                if 'user_profile' in cfg['info_size']:
                    local_model.user_profile.load_state_dict(copy.deepcopy(self.global_model.user_profile.state_dict()))
                if 'item_attr' in cfg['info_size']:
                    local_model.item_attr.load_state_dict(copy.deepcopy(self.global_model.item_attr.state_dict()))

            self.local_model_dict[i]['model'] = local_model
            self.local_test_model_dict[i]['model'] = test_model
            optimizer = make_optimizer(local_model, cfg['model_name'])
            scheduler = make_scheduler(optimizer, cfg['model_name'])

            self.local_model_dict[i]['optimizer'] = optimizer
            self.local_model_dict[i]['scheduler'] = scheduler
        return

    def update_client_parameters_with_global_parameters(self, cur_model):
        
        self.new_global_model_parameter_dict = collections.defaultdict(int)

        if cfg['federated_mode'] == 'decoder':
            cur_model = cur_model['model']          
            net_model_dict = {}

            # for key in self.global_model.state_dict():
            for key,value in self.global_model.named_parameters():
                if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key]) 
                elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                elif key in self.batch_normalization_name:     
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                else:   
                    net_model_dict[key] = copy.deepcopy(self.global_model.state_dict()[key])

            cur_model.load_state_dict(copy.deepcopy(net_model_dict), strict=False)
        else:
            raise ValueError('Not valid federated mode')
        return

    def generate_new_global_model_parameter_dict(self, model_state_dict, total_client):
        # print('total_clinet', total_client)
        if cfg['model_name'] == 'ae':
            if cfg['train_mode'] == 'private':
                # if cfg['federated_mode'] == 'decoder' and not local_parameters:
                #     # for key in self.global_model.state_dict():
                #     for key,value in self.global_model.named_parameters():
                #         if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                #             pass
                #         elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                #             pass
                #         elif key in self.batch_normalization_name:       
                #             pass
                #         else:
                #             for m in range(len(node_idx)):
                #                 cur_node_index = node_idx[m]
                #                 cur_local_model = self.load_local_model_dict(cur_node_index)['model'].state_dict()
                #                 cur_ratio = 1 / len(node_idx)
                #                 # self.global_model.state_dict()[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
                #                 # if key not in self.new_global_model_parameter_dict:
                #                 #     self.new_global_model_parameter_dict[key] = 0
                #                 self.new_global_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
                if cfg['federated_mode'] == 'all':
                    for key in model_state_dict:
                        # print('keyukeykey', key)
                        cur_ratio = 1 / total_client
                        self.new_global_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(model_state_dict[key]))

                    # for key,value in self.global_model.named_parameters():
                    #     for m in range(len(local_parameters)):
                    #         # cur_node_index = node_idx[m]
                    #         cur_local_model = local_parameters[m]
                    #         cur_ratio = 1 / len(local_parameters)
                    #         # self.global_model.state_dict()[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
                    #         # if key not in self.new_global_model_parameter_dict:
                    #         #     self.new_global_model_parameter_dict[key] = 0
                    #         self.new_global_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
        else:   
            raise ValueError('Not valid model name')
        return
    
    def combine(self, node_idx, local_parameters=None):
        
        if cfg['model_name'] == 'ae':
            if cfg['train_mode'] == 'private':
                if cfg['federated_mode'] == 'decoder' and not local_parameters:
                    # for key in self.global_model.state_dict():
                    for key,value in self.global_model.named_parameters():
                        if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                            pass
                        elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                            pass
                        elif key in self.batch_normalization_name:       
                            pass
                        else:
                            for m in range(len(node_idx)):
                                cur_node_index = node_idx[m]
                                cur_local_model = self.load_local_model_dict(cur_node_index)['model'].state_dict()
                                cur_ratio = 1 / len(node_idx)
                                # self.global_model.state_dict()[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
                                # if key not in self.new_global_model_parameter_dict:
                                #     self.new_global_model_parameter_dict[key] = 0
                                self.new_global_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
                elif cfg['federated_mode'] == 'all' and local_parameters:
                    for key,value in self.global_model.named_parameters():
                        if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                            pass
                        elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                            pass
                        elif key in self.batch_normalization_name:       
                            pass
                        else:
                            for m in range(len(local_parameters)):
                                # cur_node_index = node_idx[m]
                                cur_local_model = local_parameters[m]
                                cur_ratio = 1 / len(local_parameters)
                                # self.global_model.state_dict()[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
                                # if key not in self.new_global_model_parameter_dict:
                                #     self.new_global_model_parameter_dict[key] = 0
                                self.new_global_model_parameter_dict[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
        else:   
            raise ValueError('Not valid model name')
        return
    
    def update_local_test_model_dict(self):
        for i in range(len(self.local_test_model_dict)):

            cur_local_model_dict = self.local_model_dict[i]
            cur_model = cur_local_model_dict['model']

            cur_local_test_model_dict = self.local_test_model_dict[i]
            cur_test_model = cur_local_test_model_dict['model']
            
            net_model_dict = {}
            # for key in self.global_model.state_dict():
            for key,value in self.global_model.named_parameters():
                if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key]) 
                elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                elif key in self.batch_normalization_name:     
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                # elif 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                #     net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                else:   
                    net_model_dict[key] = copy.deepcopy(self.new_global_model_parameter_dict[key])
            cur_test_model.load_state_dict(copy.deepcopy(net_model_dict), strict=False)

        
        return

    # def update_global_model(self, client):
    #     with torch.no_grad():
    #         valid_client = [client[i] for i in range(len(client)) if client[i].active]
    #         if len(valid_client) > 0:
    #             model = eval('models.{}()'.format(cfg['model_name']))
    #             model.load_state_dict(self.model_state_dict)
    #             global_optimizer = make_optimizer(model, 'global')
    #             global_optimizer.load_state_dict(self.global_optimizer_state_dict)
    #             global_optimizer.zero_grad()
    #             weight = torch.ones(len(valid_client))
    #             weight = weight / weight.sum()
    #             for k, v in model.named_parameters():
    #                 parameter_type = k.split('.')[-1]
    #                 if 'weight' in parameter_type or 'bias' in parameter_type:
    #                     tmp_v = v.data.new_zeros(v.size())
    #                     for m in range(len(valid_client)):
    #                         tmp_v += weight[m] * valid_client[m].model_state_dict[k]
    #                     v.grad = (v.data - tmp_v).detach()
    #             global_optimizer.step()
    #             self.global_optimizer_state_dict = global_optimizer.state_dict()
    #             self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    #         for i in range(len(client)):
    #             client[i].active = False
    #     return

    def update_global_model_momentum(self):
        with torch.no_grad():
            global_optimizer = make_optimizer(self.global_model, 'global')
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
                    b = v.grad

                # for k in self.global_model.state_dict():
                #     v = self.global_model.state_dict()[k]
                #     parameter_type = k.split('.')[-1]
                #     # if 'weight' in parameter_type or 'bias' in parameter_type:
                #     tmp_v = copy.deepcopy(v.new_zeros(v.size()))
                #     for m in range(len(node_idx)):
                #         cur_node_index = node_idx[m]
                #         cur_model = self.local_model_dict[cur_node_index]['model']
                #         cur_ratio = 1 / len(node_idx)
                #         tmp_v += cur_ratio * copy.deepcopy(cur_model.state_dict()[k])
                #         # a = v.data
                #     #requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
                #     #v.data(t-1) - 1 * (v.data(t-1) - tmp_v(t)) = tmp_v(t)
                #     v.grad = (v - tmp_v).detach()
                #     # v.grad = (v.data - tmp_v).detach()
                #     b = v.grad
            global_optimizer.step()
            self.global_optimizer_state_dict = global_optimizer.state_dict()
            self.new_global_model_parameter_dict = collections.defaultdict(int)
                # self.new_global_model_parameter_dict = {k: v.cpu() for k, v in self.global_model.state_dict().items()}               
        return

    # def update_global_model_momentum(self, node_idx, local_parameters=None):
    #     with torch.no_grad():
    #         if len(node_idx) > 0:
    #             zzz = copy.deepcopy(self.global_model.state_dict())
    #             global_optimizer = make_optimizer(self.global_model, 'global')
    #             global_optimizer.load_state_dict(copy.deepcopy(self.global_optimizer_state_dict))
    #             global_optimizer.zero_grad()
    #             for k,v in self.global_model.named_parameters():
    #                 parameter_type = k.split('.')[-1]
    #                 if 'weight' in parameter_type or 'bias' in parameter_type:
    #                     tmp_v = copy.deepcopy(v.data.new_zeros(v.size()))
    #                     for m in range(len(node_idx)):
    #                         if cfg['federated_mode'] == 'decoder' and not local_parameters:
    #                             cur_node_index = node_idx[m]
    #                             cur_model_parameters = self.local_model_dict[cur_node_index]['model'].state_dict()
    #                             cur_ratio = 1 / len(node_idx)
    #                         elif cfg['federated_mode'] == 'all' and local_parameters:
    #                             cur_model_parameters = local_parameters[m]
                                
    #                         cur_ratio = 1 / len(node_idx)
    #                         tmp_v += cur_ratio * copy.deepcopy(cur_model_parameters[k])
    #                       # a = v.data
    #                     #requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
    #                     #v.data(t-1) - 1 * (v.data(t-1) - tmp_v(t)) = tmp_v(t)
    #                     v.grad = (v.data - tmp_v).detach()
    #                     b = v.grad

    #             # for k in self.global_model.state_dict():
                
    #             #     v = self.global_model.state_dict()[k]
    #             #     parameter_type = k.split('.')[-1]
    #             #     # if 'weight' in parameter_type or 'bias' in parameter_type:
    #             #     tmp_v = copy.deepcopy(v.new_zeros(v.size()))
    #             #     for m in range(len(node_idx)):
    #             #         cur_node_index = node_idx[m]
    #             #         cur_model = self.local_model_dict[cur_node_index]['model']
    #             #         cur_ratio = 1 / len(node_idx)
    #             #         tmp_v += cur_ratio * copy.deepcopy(cur_model.state_dict()[k])
    #             #         # a = v.data
    #             #     #requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
    #             #     #v.data(t-1) - 1 * (v.data(t-1) - tmp_v(t)) = tmp_v(t)
    #             #     v.grad = (v - tmp_v).detach()
    #             #     # v.grad = (v.data - tmp_v).detach()
    #             #     b = v.grad

    #             global_optimizer.step()
    #             self.global_optimizer_state_dict = global_optimizer.state_dict()
    #             # self.new_global_model_parameter_dict = {k: v.cpu() for k, v in self.global_model.state_dict().items()}               
    #     return

