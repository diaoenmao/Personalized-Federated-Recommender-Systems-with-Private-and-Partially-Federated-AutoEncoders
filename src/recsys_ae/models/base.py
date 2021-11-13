import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class Base(nn.Module):
    def __init__(self, num_users, num_items):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        if cfg['data_mode'] == 'user':
            self.register_buffer('base', torch.zeros(self.num_items))
            self.register_buffer('count', torch.zeros(self.num_items))
        elif cfg['data_mode'] == 'item':
            self.register_buffer('base', torch.zeros(self.num_users))
            self.register_buffer('count', torch.zeros(self.num_users))
        else:
            raise ValueError('Not valid data mode')
    def forward(self, input):
        output = {}
        if cfg['data_mode'] == 'user':
            if cfg['target_mode'] == 'explicit':
                if self.training:
                    self.base.scatter_add_(0, input['item'], input['rating'])
                    self.count.scatter_add_(0, input['item'], input['rating'].new_ones(input['rating'].size()))
                output['target_rating'] = self.base[input['target_item']] / (self.count[input['target_item']] + 1e-10)
                output['target_rating'][self.count[input['target_item']] == 0] = (self.base[self.count != 0] /
                                                                                  self.count[self.count != 0]).mean()
                output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
            elif cfg['target_mode'] == 'implicit':
                if self.training:
                    self.base.scatter_add_(0, input['item'], input['rating'])
                    self.count = self.count + torch.unique(input['user']).size(0)
                output['target_rating'] = self.base[input['target_item']] / self.count[input['target_item']]
                output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
            else:
                raise ValueError('Not valid target mode')
        elif cfg['data_mode'] == 'item':
            if cfg['target_mode'] == 'explicit':
                if self.training:
                    self.base.scatter_add_(0, input['user'], input['rating'])
                    self.count.scatter_add_(0, input['user'], input['rating'].new_ones(input['rating'].size()))
                output['target_rating'] = self.base[input['target_user']] / (self.count[input['target_user']] + 1e-10)
                output['target_rating'][self.count[input['target_user']] == 0] = (self.base[self.count != 0] /
                                                                                  self.count[self.count != 0]).mean()
                output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
            elif cfg['target_mode'] == 'implicit':
                if self.training:
                    self.base.scatter_add_(0, input['user'], input['rating'])
                    self.count = self.count + torch.unique(input['item']).size(0)
                output['target_rating'] = self.base[input['target_user']] / self.count[input['target_user']]
                output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
            else:
                raise ValueError('Not valid target mode')
        else:
            raise ValueError('Not valid data mode')
        return output


def base(num_users=None, num_items=None):
    num_users = cfg['num_users']['data'] if num_users is None else num_users
    num_items = cfg['num_items']['data'] if num_items is None else num_items
    model = Base(num_users, num_items)
    return model
