import copy
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
        self.global_model = None
        self.batch_normalization_name = {'encoder.blocks.1.weight', 'encoder.blocks.1.bias', 'encoder.blocks.1.running_mean', 'encoder.blocks.1.running_var', 'encoder.blocks.1.num_batches_tracked'}
        # self.batch_normalization_name = {}
        
    def reset_parameters(self, model):
        for model in [model.encoder, model.decoder]:
            for m in model.blocks:
                # if item m is nn.Linear, set its value to xavier_uniform heuristic value
                # if isinstance(m, nn.Linear):
                if not isinstance(m, nn.Tanh):
                    m.weight.data.zero_()
                    if m.bias is not None:
                        # set bias to 0
                        m.bias.data.zero_()
        

    # def reset_global_model(self):
    #     self.global_model = copy.deepcopy(self.example_model)
    #     self.reset_parameters(self.global_model)

    def get_global_encoder_model_parameters(self):
        return self.global_model.encoder.state_dict()
    
    def get_global_decoder_model_parameters(self):
        return self.global_model.decoder.state_dict()

    def create_local_model_dict(self):
        cur_num_users = self.data_split_info[0]['num_users']
        cur_num_items = self.data_split_info[0]['num_items']
        self.global_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
            
        a = self.global_model.state_dict()
        for i in range(cfg['num_nodes']):

            cur_num_users = self.data_split_info[i]['num_users']
            cur_num_items = self.data_split_info[i]['num_items']
            local_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
            
            test_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))

            if cfg['federated_mode'] == 'encoder':
                local_model.encoder.load_state_dict(copy.deepcopy(self.global_model.encoder.state_dict()))
            elif cfg['federated_mode'] == 'decoder':
                local_model.decoder.load_state_dict(copy.deepcopy(self.global_model.decoder.state_dict()))
            elif cfg['federated_mode'] == 'all':
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

    def load_local_model_dict(self, index):
        return self.local_model_dict[index]

    def load_local_test_model_dict(self, index):
        return self.local_test_model_dict[index]

    def update_client_parameters_with_global_parameters(self, index):
       
        cur_local_model_dict = self.local_model_dict[index]
        cur_model = cur_local_model_dict['model']
        
        a = self.global_model.named_parameters()
        a_list = []
        for key in a:
            a_list.append(key)
        zz_list = []
        for k,v in self.global_model.keys():
            zz_list.append(k)
        b = self.global_model.state_dict()
        net_model_dict = {}
        for key, value in self.global_model.named_parameters():
            if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key]) 
            elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
            elif key in self.batch_normalization_name:     
                net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
            else:   
                net_model_dict[key] = copy.deepcopy(self.global_model.state_dict()[key])
        cur_model.load_state_dict(copy.deepcopy(net_model_dict))

        return


    def combine_and_update_global_parameters(self, node_idx, participated_user):
        
        if cfg['model_name'] == 'ae':
            total_participated_user = sum(participated_user)
            self.reset_parameters(self.global_model)
            if cfg['train_mode'] == 'private':
                
                for key, value in self.global_model.named_parameters():
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
                            cur_ratio = participated_user[m] / total_participated_user
                            self.global_model.state_dict()[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))                
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
            for key, value in self.global_model.named_parameters():
                if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key]) 
                elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                elif key in self.batch_normalization_name:     
                    net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
                else:   
                    net_model_dict[key] = copy.deepcopy(self.global_model.state_dict()[key])
            cur_test_model.load_state_dict(copy.deepcopy(net_model_dict))

        return

    # def update_local_model_dict(self):
    #     if cfg['model_name'] == 'ae':
    #         if cfg['train_mode'] == 'private':
    #             for i in range(len(self.local_model_dict)):
    #                 cur_local_model_dict = self.local_model_dict[i]
    #                 cur_model = cur_local_model_dict['model']
    #                 # cur_model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
                    
    #                 net_model_dict = {}
    #                 for key, value in self.global_model.named_parameters():
    #                     if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
    #                         net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key]) 
    #                     elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
    #                         net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
    #                     elif key in self.batch_normalization_name:     
    #                         net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
    #                     else:   
    #                         net_model_dict[key] = copy.deepcopy(self.global_model.state_dict()[key])
    #                 cur_model.load_state_dict(copy.deepcopy(net_model_dict))
    #     else:
    #         raise ValueError('Not valid model name')
        
    #     return 


# class Federation:
#     def __init__(self, data_split_info):
#         # 清零
#         self.data_split_info = data_split_info
        
#         self.local_model_dict = collections.defaultdict(dict)
#         self.example_model = None
#         self.global_model = None
#         self.batch_normalization_name = {}
#         # self.batch_normalization_name = {'encoder.blocks.1.weight', 'encoder.blocks.1.bias', 'encoder.blocks.4.weight', 'encoder.blocks.4.bias'}
        
#     def reset_parameters(self, model):

#         for model in [model.encoder, model.decoder]:
#             for m in model.blocks:
#                 # if item m is nn.Linear, set its value to xavier_uniform heuristic value
#                 if isinstance(m, nn.Linear):
#                 # if not isinstance(m, nn.Tanh):
#                     m.weight.data.zero_()
#                     if m.bias is not None:
#                         # set bias to 0
#                         m.bias.data.zero_()
        

#     def new_global_model(self):
#         self.global_model = copy.deepcopy(self.example_model)
#         self.reset_parameters(self.global_model)

#     def get_global_encoder_model_parameters(self):
#         return self.global_model.encoder.state_dict()
    
#     def get_global_decoder_model_parameters(self):
#         return self.global_model.decoder.state_dict()

#     def create_local_model_dict(self):
#         cur_num_users = self.data_split_info[0]['num_users']
#         cur_num_items = self.data_split_info[0]['num_items']
#         self.example_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
#                 'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
#         a = self.example_model.state_dict()
#         # a = self.example_model.state_dict()
#         for i in range(cfg['num_nodes']):

#             cur_num_users = self.data_split_info[i]['num_users']
#             cur_num_items = self.data_split_info[i]['num_items']
#             local_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
#                 'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
            
#             if cfg['federated_mode'] == 'encoder':
#                 local_model.encoder.load_state_dict(copy.deepcopy(self.example_model.encoder.state_dict()))
#             elif cfg['federated_mode'] == 'decoder':
#                 local_model.decoder.load_state_dict(copy.deepcopy(self.example_model.decoder.state_dict()))
#             elif cfg['federated_mode'] == 'all':
#                 local_model.encoder.load_state_dict(copy.deepcopy(self.example_model.encoder.state_dict()))
#                 local_model.decoder.load_state_dict(copy.deepcopy(self.example_model.decoder.state_dict()))
                
#             if cfg['info_size'] is not None:
#                 if 'user_profile' in cfg['info_size']:
#                     local_model.user_profile.load_state_dict(copy.deepcopy(self.example_model.user_profile.state_dict()))
#                 if 'item_attr' in cfg['info_size']:
#                     local_model.item_attr.load_state_dict(copy.deepcopy(self.example_model.item_attr.state_dict()))


#             self.local_model_dict[i]['model'] = local_model
#             optimizer = make_optimizer(local_model, cfg['model_name'])
#             scheduler = make_scheduler(optimizer, cfg['model_name'])

#             self.local_model_dict[i]['optimizer'] = optimizer
#             self.local_model_dict[i]['scheduler'] = scheduler
#         return

#     def load_local_model_dict(self, index):
#         return self.local_model_dict[index]

#     def combine(self, node_idx, participated_user):
        
#         if cfg['model_name'] == 'ae':
#             total_participated_user = sum(participated_user)
#             if cfg['train_mode'] == 'private':
#                 # for key, value in self.global_model.named_parameters():
#                 #     for m in range(len(node_idx)):
#                 #         cur_node_index = node_idx[m]
#                 #         cur_local_model = self.load_local_model_dict(cur_node_index)['model'].state_dict()
#                 #         cur_ratio = participated_user[m] / total_participated_user
#                 #         self.global_model.state_dict()[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))
                
#                 for key, value in self.global_model.named_parameters():
#                     if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
#                         pass
#                     elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
#                         pass
#                     elif key in self.batch_normalization_name:       
#                         pass
#                     else:
#                         for m in range(len(node_idx)):
#                             cur_node_index = node_idx[m]
#                             cur_local_model = self.load_local_model_dict(cur_node_index)['model'].state_dict()
#                             cur_ratio = participated_user[m] / total_participated_user
#                             self.global_model.state_dict()[key] += (cur_ratio * copy.deepcopy(cur_local_model[key]))                
#         else:   
#             raise ValueError('Not valid model name')
#         return
    
#     def update_local_model_dict(self):
#         if cfg['model_name'] == 'ae':
#             if cfg['train_mode'] == 'private':
#                 for i in range(len(self.local_model_dict)):
#                     cur_local_model_dict = self.local_model_dict[i]
#                     cur_model = cur_local_model_dict['model']
#                     # cur_model.load_state_dict(copy.deepcopy(self.global_model.state_dict()))
                    
#                     net_model_dict = {}
#                     for key, value in self.global_model.named_parameters():
#                         if cfg['federated_mode'] == 'encoder' and 'decoder' in key:
#                             net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key]) 
#                         elif cfg['federated_mode'] == 'decoder' and 'encoder' in key:
#                             net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
#                         elif key in self.batch_normalization_name:     
#                             net_model_dict[key] = copy.deepcopy(cur_model.state_dict()[key])
#                         else:   
#                             net_model_dict[key] = copy.deepcopy(self.global_model.state_dict()[key])
#                     cur_model.load_state_dict(copy.deepcopy(net_model_dict))
#         else:
#             raise ValueError('Not valid model name')
        
#         return 
