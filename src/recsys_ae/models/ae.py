import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import loss_fn
from config import cfg


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        blocks = [nn.Linear(input_size, hidden_size[0]),
                  nn.Tanh()]
        for i in range(len(hidden_size) - 1):
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.blocks(x)
        return x


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        blocks = []
        for i in range(len(hidden_size) - 1):
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        blocks.append(nn.Linear(hidden_size[-1], output_size))
        self.blocks = nn.Sequential(*blocks)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.blocks:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        return

    def forward(self, x):
        x = self.blocks(x)
        return x


class AE(nn.Module):
    def __init__(self, encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
                 decoder_hidden_size, info_size):
        super().__init__()
        self.info_size = info_size
        if cfg['data_mode'] == 'user':
            self.encoder = Encoder(encoder_num_items, encoder_hidden_size)
            self.decoder = Decoder(decoder_num_items, decoder_hidden_size)
        elif cfg['data_mode'] == 'item':
            self.encoder = Encoder(encoder_num_users, encoder_hidden_size)
            self.decoder = Decoder(decoder_num_users, decoder_hidden_size)
        else:
            raise ValueError('Not valid data mode')
        self.dropout = nn.Dropout(p=0.5)
        if info_size is not None:
            if 'user_profile' in info_size:
                self.user_profile = Encoder(info_size['user_profile'], encoder_hidden_size)
            if 'item_attr' in info_size:
                self.item_attr = Encoder(info_size['item_attr'], encoder_hidden_size)

    def forward(self, input):
        output = {}
        with torch.no_grad():
            if cfg['data_mode'] == 'user':
                user, user_idx = torch.unique(torch.cat([input['user'], input['target_user']]), return_inverse=True)
                num_users = len(user)
                rating = torch.zeros((num_users, self.encoder.input_size), device=user.device)
                rating[user_idx[:len(input['user'])], input['item']] = input['rating']
                input['rating'] = rating
                rating = torch.full((num_users, self.decoder.output_size), float('nan'), device=user.device)
                rating[user_idx[len(input['user']):], input['target_item']] = input['target_rating']
                input['target_rating'] = rating
            elif cfg['data_mode'] == 'item':
                item, item_idx = torch.unique(torch.cat([input['item'], input['target_item']]), return_inverse=True)
                num_items = len(item)
                rating = torch.zeros((num_items, self.encoder.input_size), device=item.device)
                rating[item_idx[:len(input['item'])], input['user']] = input['rating']
                input['rating'] = rating
                rating = torch.full((num_items, self.decoder.output_size), float('nan'), device=item.device)
                rating[item_idx[len(input['item']):], input['target_user']] = input['target_rating']
                input['target_rating'] = rating
        x = input['rating']
        encoded = self.encoder(x)
        if self.info_size is not None:
            if 'user_profile' in input:
                user_profile = input['user_profile']
                user_profile = self.user_profile(user_profile)
                encoded = encoded + user_profile
            if 'item_attr' in input:
                item_attr = input['item_attr']
                item_attr = self.item_attr(item_attr)
                encoded = encoded + item_attr
        code = self.dropout(encoded)
        decoded = self.decoder(code)
        output['target_rating'] = decoded
        target_mask = ~(input['target_rating'].isnan())
        output['target_rating'], input['target_rating'] = output['target_rating'][target_mask], input['target_rating'][
            target_mask]
        if 'local' in input and input['local']:
            output['loss'] = F.mse_loss(output['target_rating'], input['target_rating'])
        else:
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])
        return output


def ae(encoder_num_users=None, encoder_num_items=None, decoder_num_users=None, decoder_num_items=None):
    encoder_num_users = cfg['num_users']['data'] if encoder_num_users is None else encoder_num_users
    encoder_num_items = cfg['num_items']['data'] if encoder_num_items is None else encoder_num_items
    decoder_num_users = cfg['num_users']['target'] if decoder_num_users is None else decoder_num_users
    decoder_num_items = cfg['num_items']['target'] if decoder_num_items is None else decoder_num_items
    encoder_hidden_size = cfg['ae']['encoder_hidden_size']
    decoder_hidden_size = cfg['ae']['decoder_hidden_size']
    info_size = cfg['info_size']
    model = AE(encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
               decoder_hidden_size, info_size)
    return model
