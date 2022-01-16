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
        self.example_model = None
        self.global_encoder_model = None
        self.global_decoder_model = None
        
    def reset_parameters(self, model):
        for m in model.blocks:
            # if item m is nn.Linear, set its value to xavier_uniform heuristic value
            if isinstance(m, nn.Linear):
                m.weight.data.zero_()
                if m.bias is not None:
                    # set bias to 0
                    m.bias.data.zero_()

    def new_global_model(self):
        self.global_encoder_model = copy.deepcopy(self.example_model.encoder)
        self.reset_parameters(self.global_encoder_model)
        self.global_decoder_model = copy.deepcopy(self.example_model.decoder)
        self.reset_parameters(self.global_decoder_model)

    def get_global_encoder_model_parameters(self):
        return self.global_encoder_model.state_dict()
    
    def get_global_decoder_model_parameters(self):
        return self.global_decoder_model.state_dict()

    def create_local_model_dict(self):
        cur_num_users = self.data_split_info[0]['num_users']
        cur_num_items = self.data_split_info[0]['num_items']
        self.example_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))

        for i in range(cfg['num_nodes']):

            cur_num_users = self.data_split_info[i]['num_users']
            cur_num_items = self.data_split_info[i]['num_items']
            local_model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))

            local_model.decoder.load_state_dict(copy.deepcopy(self.example_model.decoder.state_dict()))
            if cfg['federated_mode'] == 'all':
                local_model.encoder.load_state_dict(copy.deepcopy(self.example_model.encoder.state_dict()))

            if cfg['info_size'] is not None:
                if 'user_profile' in cfg['info_size']:
                    local_model.user_profile.load_state_dict(copy.deepcopy(self.example_model.user_profile.state_dict()))
                if 'item_attr' in cfg['info_size']:
                    local_model.item_attr.load_state_dict(copy.deepcopy(self.example_model.item_attr.state_dict()))


            self.local_model_dict[i]['model'] = local_model
            # root = processed_folder(i, True)
            # model_path = os.path.join(root, cfg['model_name'] + '.pt')
            # save(model, model_path)

            optimizer = make_optimizer(local_model, cfg['model_name'])
            scheduler = make_scheduler(optimizer, cfg['model_name'])

            self.local_model_dict[i]['optimizer'] = optimizer
            self.local_model_dict[i]['scheduler'] = scheduler
        return

    def load_local_model_dict(self, index):
        return self.local_model_dict[index]

    def combine(self, local_parameters, participated_user):
        
        if cfg['model_name'] == 'ae':
            total_participated_user = sum(participated_user)
            if cfg['train_mode'] == 'private':
                for key, value in self.global_decoder_model.named_parameters():
                    for m in range(len(local_parameters)):
                        local_parameters_key = 'decoder.' + key
                        cur_ratio = participated_user[m] / total_participated_user
                        self.global_decoder_model.state_dict()[key] += (cur_ratio * copy.deepcopy(local_parameters[m][local_parameters_key]))
                        # self.global_decoder_model.state_dict()[key] += (copy.deepcopy(local_parameters[m][local_parameters_key]) / len(local_parameters))
            if cfg['federated_mode'] == 'all':
                for key, value in self.global_encoder_model.named_parameters():
                    for m in range(len(local_parameters)):
                        local_parameters_key = 'encoder.' + key
                        cur_ratio = participated_user[m] / total_participated_user
                        self.global_encoder_model.state_dict()[key] += (cur_ratio * copy.deepcopy(local_parameters[m][local_parameters_key]))
                        # self.global_encoder_model.state_dict()[key] += (copy.deepcopy(local_parameters[m][local_parameters_key]) / len(local_parameters))
        else:   
            raise ValueError('Not valid model name')
        
        return
    
    def update_local_model_dict(self):

        if cfg['model_name'] == 'ae':
            if cfg['train_mode'] == 'private':
                for i in range(len(self.local_model_dict)):
                    cur_local_model_dict = self.local_model_dict[i]
                    cur_model = cur_local_model_dict['model']
                    cur_model.decoder.load_state_dict(copy.deepcopy(self.global_decoder_model.state_dict()))
                
            if cfg['federated_mode'] == 'all':
                for i in range(len(self.local_model_dict)):
                    cur_local_model_dict = self.local_model_dict[i]
                    cur_model = cur_local_model_dict['model']
                    cur_model.encoder.load_state_dict(copy.deepcopy(self.global_encoder_model.state_dict()))
        else:
            raise ValueError('Not valid model name')
        
        return 
