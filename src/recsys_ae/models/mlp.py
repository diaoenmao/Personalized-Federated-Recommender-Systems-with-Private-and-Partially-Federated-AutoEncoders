import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class MLP(nn.Module):
    def __init__(self, num_users, num_items, hidden_size, info_size):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.hidden_size = hidden_size
        self.info_size = info_size
        self.user_weight = nn.Embedding(num_users, hidden_size[0])
        self.item_weight = nn.Embedding(num_items, hidden_size[0])
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        if self.info_size is not None:
            if 'user_profile' in info_size:
                self.user_profile = nn.Linear(info_size['user_profile'], hidden_size[0])
            if 'item_attr' in info_size:
                self.item_attr = nn.Linear(info_size['item_attr'], hidden_size[0])
        fc = []
        for i in range(len(hidden_size) - 1):
            if i == 0:
                input_size = 2 * hidden_size[i]
                if self.info_size is not None:
                    if 'user_profile' in info_size:
                        input_size = input_size + hidden_size[i]
                    if 'item_attr' in info_size:
                        input_size = input_size + hidden_size[i]
            else:
                input_size = hidden_size[i]
            fc.append(torch.nn.Linear(input_size, hidden_size[i + 1]))
            fc.append(nn.ReLU())
        self.fc = nn.Sequential(*fc)
        self.affine = nn.Linear(hidden_size[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_weight.weight, 0.0, 0.01)
        nn.init.normal_(self.item_weight.weight, 0.0, 0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.affine.bias)
        return

    def user_embedding(self, user):
        embedding = self.user_weight(user) + self.user_bias(user)
        return embedding

    def item_embedding(self, item):
        embedding = self.item_weight(item) + self.item_bias(item)
        return embedding

    def forward(self, input):
        output = {}
        if self.training:
            user = input['user']
            item = input['item']
            rating = input['rating']
            if self.info_size is not None:
                user_profile = input['user_profile'] if 'user_profile' in input else None
                item_attr = input['item_attr'] if 'item_attr' in input else None
            else:
                user_profile = None
                item_attr = None
        else:
            user = input['target_user']
            item = input['target_item']
            rating = input['target_rating']
            if self.info_size is not None:
                user_profile = input['target_user_profile'] if 'target_user_profile' in input else None
                item_attr = input['target_item_attr'] if 'target_item_attr' in input else None
            else:
                user_profile = None
                item_attr = None
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        mlp = torch.cat([user_embedding, item_embedding], dim=-1)
        if self.info_size is not None:
            info = torch.tensor([], device=user.device)
            if user_profile is not None:
                user_profile = self.user_profile(user_profile)
                info = torch.cat([info, user_profile], dim=-1)
            if item_attr is not None:
                item_attr = self.item_attr(item_attr)
                info = torch.cat([info, item_attr], dim=-1)
            mlp = torch.cat([mlp, info], dim=-1)
        mlp = self.fc(mlp)
        output['target_rating'] = self.affine(mlp).view(-1)
        output['loss'] = loss_fn(output['target_rating'], rating)
        return output


def mlp(num_users=None, num_items=None):
    num_users = cfg['num_users']['data'] if num_users is None else num_users
    num_items = cfg['num_items']['data'] if num_items is None else num_items
    hidden_size = cfg['mlp']['hidden_size']
    info_size = cfg['info_size']
    model = MLP(num_users, num_items, hidden_size, info_size)
    return model
