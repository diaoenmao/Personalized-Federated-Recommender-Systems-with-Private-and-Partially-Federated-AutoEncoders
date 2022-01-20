import os 
import math
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import load, processed_folder
from .models_utils import loss_fn
from config import cfg


class Encoder(nn.Module):

    """
    Initialize Encoder.

    Parameters:
        input_size - Integer. The number of items. ex. 1682.
        hidden_size - List[Integer]. ex. [256, 128]. The size of neural network of encoder.

    Returns:
        Instance of class Encoder.

    Raises:
        None
    """

    def __init__(self, input_size, hidden_size):
        # Construct Neural network
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        blocks = []

        # Put the input and the first layer into blocks(list).
        # Put the nn.Tanh(), which is an activation function, into the blocks
       
        blocks.append(nn.Linear(input_size, hidden_size[0]))
        # blocks.append(nn.LayerNorm(hidden_size[0]))
        # blocks.append(nn.BatchNorm1d(hidden_size[0]))
        blocks.append(nn.Tanh())
        
        # Put the rest layers in hidden_size and activation function into the blocks
        # Set range to len(hidden_size)-1 to avoid overflow of index
        for i in range(len(hidden_size) - 1):
            # blocks.append(nn.LayerNorm(hidden_size[i]))
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            # blocks.append(nn.LayerNorm(hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        
        # nn.Sequential: A sequential container. 
        # Modules will be added to it in the order they are passed in the constructor.
        # * is iterable unpacking notation in Python
        self.blocks = nn.Sequential(*blocks)

        # set initial parameters
        self.reset_parameters()

    def reset_parameters(self):
        # if ('Encoder_instance' not in cfg or 'user_profile_Encoder_instance' not in cfg 
        #     or 'item_attr_Encoder_instance' not in cfg):
        # if 'Encoder_instance' not in cfg:
        for m in self.blocks:
            # if item m is nn.Linear, set its value to xavier_uniform heuristic value
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # set bias to 0
                    m.bias.data.zero_()
        return

    def forward(self, x):
        # pass the parameter x into the self.blocks (model)
        # get the output of the model
        x = self.blocks(x)
        return x


class Decoder(nn.Module):

    """
    Initialize Decoder.

    Parameters:
        output_size - Integer. The number of items. ex. 1682.
        hidden_size - List[Integer]. ex. [128, 256]. The size of neural network of decoder.

    Returns:
        Instance of class Decoder.

    Raises:
        None
    """

    def __init__(self, output_size, hidden_size):
        # Construct Neural network
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        blocks = []

        # Put the layers in hidden_size and activation function into the blocks
        # Set range to len(hidden_size)-1 to avoid overflow of index
        # blocks.append(nn.LayerNorm(hidden_size[0]))
        for i in range(len(hidden_size) - 1):
            # blocks.append(nn.LayerNorm(hidden_size[i]))
            blocks.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            # blocks.append(nn.LayerNorm(hidden_size[i + 1]))
            blocks.append(nn.Tanh())
        
        # Put the last layer and the output into blocks(list).
        # Put the nn.Tanh(), which is an activation function, into the blocks
        # blocks.append(nn.LayerNorm(hidden_size[-1]))
        blocks.append(nn.Linear(hidden_size[-1], output_size))

        # nn.Sequential: A sequential container. 
        # Modules will be added to it in the order they are passed in the constructor.
        # * is iterable unpacking notation in Python
        self.blocks = nn.Sequential(*blocks)

        # set initial parameters
        self.reset_parameters()

    def reset_parameters(self):
        # if 'Decoder_instance' not in cfg:
        for m in self.blocks:
            # if item m is nn.Linear, set its value to xavier_uniform heuristic value
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # set bias to 0
                    m.bias.data.zero_()
        return

    def forward(self, x):
        # pass the parameter x into the self.blocks (model)
        # get the output of the model
        x = self.blocks(x)
        return x


class AE(nn.Module):

    """
    1. Initialize the autoencoder + autodecoder system.
    2. calculate the system output, loss in forward() and return output

    Parameters:
        encoder_num_users - Integer. ex. 943 
        encoder_num_items - Integer. ex. 1682
        decoder_num_users - Integer. ex. 943
        decoder_num_items - Integer. ex. 1682
        encoder_hidden_size - List[Integer]. ex. [256, 128]. The size of neural network of encoder.
        decoder_hidden_size - List[Integer]. ex. [128, 256]. The size of neural network of decoder.
        info_size - Dict. {'user_profile': 30, 'item_attr': 18}

    Returns:
        Instance of class AE

    Raises:
        None
    """

    def __init__(self, encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
                 decoder_hidden_size, info_size):
        super().__init__()
        self.info_size = info_size
        if cfg['data_mode'] == 'user':
            # Initialize Encoder and Decoder
            self.encoder = Encoder(encoder_num_items, encoder_hidden_size)
            self.decoder = Decoder(decoder_num_items, decoder_hidden_size)
            # a = cfg['Encoder_instance'].state_dict()
            # b = cfg['Decoder_instance'].state_dict()
            # c = self.encoder.state_dict()
            # d = self.decoder.state_dict()

            # self.encoder.load_state_dict(copy.deepcopy(cfg['Encoder_instance'].state_dict()))
            # self.decoder.load_state_dict(copy.deepcopy(cfg['Decoder_instance'].state_dict()))
            # e = self.encoder.state_dict()
            # f = self.decoder.state_dict()
        # elif cfg['data_mode'] == 'item':
        #     self.encoder = Encoder(encoder_num_users, encoder_hidden_size)
        #     self.decoder = Decoder(decoder_num_users, decoder_hidden_size)
        else:
            raise ValueError('Not valid data mode')

        # set dropout of the model
        self.dropout = nn.Dropout(p=0.5)

        # If we set info_size, we need to take the side information into account
        if info_size is not None:
            if 'user_profile' in info_size:
                self.user_profile = Encoder(info_size['user_profile'], encoder_hidden_size)
                # self.user_profile.load_state_dict(copy.deepcopy(cfg['user_profile_Encoder_instance'].state_dict()))
            if 'item_attr' in info_size:
                self.item_attr = Encoder(info_size['item_attr'], encoder_hidden_size)
                # self.item_attr.load_state_dict(copy.deepcopy(cfg['item_attr_Encoder_instance'].state_dict()))

    def forward(self, input):
        output = {}
        # torch.no_grad(): Context-manager that disabled gradient calculation.
        with torch.no_grad():
            if cfg['data_mode'] == 'user':
                # get the unique user of batch data, which should equal to batch size
                user, user_idx = torch.unique(torch.cat([input['user'], input['target_user']]), return_inverse=True)
                num_users = len(user)
                rating = torch.zeros((num_users, self.encoder.input_size), device=user.device)
                # a = len(input['user'])
                # b = user_idx[:len(input['user'])]
                rating[ user_idx[:len(input['user'])], input['item'] ] = input['rating']
                input['rating'] = rating
                rating = torch.full((num_users, self.decoder.output_size), float('nan'), device=user.device)
                rating[user_idx[len(input['user']):], input['target_item']] = input['target_rating']
                input['target_rating'] = rating
                # torch.set_printoptions(threshold=np.inf)
                # print('rating.item()', rating)
            # elif cfg['data_mode'] == 'item':
            #     item, item_idx = torch.unique(torch.cat([input['item'], input['target_item']]), return_inverse=True)
            #     num_items = len(item)
            #     rating = torch.zeros((num_items, self.encoder.input_size), device=item.device)
            #     rating[item_idx[:len(input['item'])], input['user']] = input['rating']
            #     input['rating'] = rating
            #     rating = torch.full((num_items, self.decoder.output_size), float('nan'), device=item.device)
            #     rating[item_idx[len(input['item']):], input['target_user']] = input['target_rating']
            #     input['target_rating'] = rating
        

        # use input['rating'] as input and pass it to self.encoder (Encoder)
        # in self.encoder: __call__() => forward()
        x = input['rating']
        encoded = self.encoder(x)

        if self.info_size is not None:
            if 'user_profile' in input:
                # Use input['user_profile'] as input and pass it to self.user_profile (Encoder)
                user_profile = input['user_profile']
                user_profile = self.user_profile(user_profile)
                # add result from self.user_profile (Encoder) to basic encoder result
                encoded = encoded + user_profile
            if 'item_attr' in input:
                # Use input['item_attr'] as input and pass it to self.item_attr (Encoder)
                item_attr = input['item_attr']
                item_attr = self.item_attr(item_attr)
                # add result from self.item_attr (Encoder) to basic encoder result
                encoded = encoded + item_attr
        
        # dropout the encoder result
        code = self.dropout(encoded)
        
        # pass the encoder result to decoder
        # in self.decoder: __call__() => forward()
        decoded = self.decoder(code)

        # handle output
        output['target_rating'] = decoded
        # generate boolean matrix to indicate if the input['target_rating'][x][y] is infinite
        # a = input['target_rating']
        target_mask = ~(input['target_rating'].isnan())
        output['target_rating'], input['target_rating'] = output['target_rating'][target_mask], input['target_rating'][
            target_mask]
            
        if 'local' in input and input['local']:
            output['loss'] = F.mse_loss(output['target_rating'], input['target_rating'])
        else:
            output['loss'] = loss_fn(output['target_rating'], input['target_rating'])

        return output


def ae(encoder_num_users=None, encoder_num_items=None, decoder_num_users=None, decoder_num_items=None):
    
    """
    1. Handle some parameters
    2. call class AE

    Parameters:
        encoder_num_users - Integer. Default is None.  
        encoder_num_items - Integer. Default is None. 
        decoder_num_users - Integer. Default is None. 
        decoder_num_items - Integer. Default is None. 

    Returns:
        model - object. Instance of class AE

    Raises:
        None
    """
    
    encoder_num_users = cfg['num_users']['data'] if encoder_num_users is None else encoder_num_users
    encoder_num_items = cfg['num_items']['data'] if encoder_num_items is None else encoder_num_items
    decoder_num_users = cfg['num_users']['target'] if decoder_num_users is None else decoder_num_users
    decoder_num_items = cfg['num_items']['target'] if decoder_num_items is None else decoder_num_items

    # hidden_size is defined in the utils.py / process_control()
    encoder_hidden_size = cfg['ae']['encoder_hidden_size']
    decoder_hidden_size = cfg['ae']['decoder_hidden_size']
    info_size = cfg['info_size']

    # if cfg['data_mode'] == 'user':
    #     if 'Encoder_instance' not in cfg:
    #         cfg['Encoder_instance'] = Encoder(encoder_num_items, encoder_hidden_size)
    #     if 'Decoder_instance' not in cfg:
    #         cfg['Decoder_instance'] = Decoder(decoder_num_items, decoder_hidden_size)
    # #     if info_size and 'user_profile' in info_size and 'user_profile_Encoder_instance' not in cfg:
    #         cfg['user_profile_Encoder_instance'] = Encoder(info_size['user_profile'], encoder_hidden_size)
    #     if info_size and 'item_attr' in info_size and 'item_attr_Encoder_instance' not in cfg:
    #         cfg['item_attr_Encoder_instance'] = Encoder(info_size['item_attr'], encoder_hidden_size)

    model = AE(encoder_num_users, encoder_num_items, decoder_num_users, decoder_num_items, encoder_hidden_size,
               decoder_hidden_size, info_size)
    return model
