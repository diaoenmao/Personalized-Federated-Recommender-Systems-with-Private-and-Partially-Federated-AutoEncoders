import torch
import torch.nn as nn
from .utils import loss_fn
from config import cfg


class Assist(nn.Module):
    def __init__(self, ar, ar_mode, num_organizations, aw_mode):
        super().__init__()
        self.ar_mode = ar_mode
        self.aw_mode = aw_mode
        if self.ar_mode == 'optim':
            self.assist_rate = nn.Parameter(torch.tensor(ar))
        elif self.ar_mode == 'constant':
            self.register_buffer('assist_rate', torch.tensor(ar))
        else:
            raise ValueError('Not valid ar mode')
        if self.aw_mode == 'optim':
            self.assist_weight = nn.Parameter(torch.ones(num_organizations) / num_organizations)
        elif self.aw_mode == 'constant':
            self.register_buffer('assist_weight', torch.ones(num_organizations) / num_organizations)
        else:
            raise ValueError('Not valid aw mode')

    def forward(self, input):
        output = {}
        output['target'] = input['history'] + self.assist_rate * (input['output'] *
                                                                  self.assist_weight.softmax(-1)).sum(-1)
        if 'target' in input:
            output['loss'] = loss_fn(output['target'], input['target'])
        return output


def assist():
    ar = cfg['assist']['ar']
    ar_mode = cfg['assist']['ar_mode']
    num_organizations = cfg['num_organizations']
    aw_mode = cfg['assist']['aw_mode']
    model = Assist(ar, ar_mode, num_organizations, aw_mode)
    return model
