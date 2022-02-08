import argparse
import copy
import datetime
from platform import node
from numpy import mod

from torch.optim import optimizer
import models
import os
import shutil
import time
import torch
import copy
import gc
import sys
import math
import random
import numpy as np
import collections
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from metrics import Metric
from fed import Federation
from utils import processed_folder, save, load, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
# create parser
parser = argparse.ArgumentParser(description='cfg')
# use add_argument() to add the value in yaml to parser
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
# add a new key (--control_name) in parser, value is None
parser.add_argument('--control_name', default=None, type=str)
# vars() returns the dict object of the key:value (typed in by the user) of parser.parse_args(). args now is dict 
args = vars(parser.parse_args())
# Updata the cfg using args in helper function => config.py / process_args(args)
process_args(args)


def main():
    # utils.py / process_control()
    # disassemble cfg['control']
    # add the model parameter
    process_control()
    # Get all integer from cfg['init_seen'] to cfg['init_seed'] + cfg['num_experiments'] - 1
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        # (seens[i] + cfg['control_name']) as experiment label
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))

        # Run experiment
        runExperiment()
    return

def runExperiment():

    # get seed and set the seed to CPU and GPU
    # same seed gives same result
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    
    # data.py / fetch_dataset(ML100K)
    # dataset is a dict, has 2 keys - 'train', 'test'
    # dataset['train'] is the instance of corresponding dataset class
    # 一整个 =》 分开
    dataset = fetch_dataset(cfg['data_name'])

    # utils.py / process_dataset(dataset)
    # add some key:value (size, num) to cfg
    process_dataset(dataset)

    # resume

    # if data_split is None:
    data_split, data_split_info = split_dataset(dataset, cfg['num_nodes'], cfg['data_split_mode'])

    # data.py / make_data_loader(dataset)
    # data_loader is a dict, has 2 keys - 'train', 'test'
    # data_loader['train'] is the instance of DataLoader (class in PyTorch), which is iterable (可迭代对象)
    # data_loader = make_data_loader(dataset)

    # models / cfg['model_name'].py initializes the model, for example, models / ae.py / class AE
    # .to(cfg["device"]) means copy the tensor to the specific GPU or CPU, and run the 
    # calculation there.
    # model is the instance of class AE (in models / ae.py). It contains the training process of 
    #   Encoder and Decoder.
    federation = Federation(data_split_info)

    if cfg['target_mode'] == 'explicit':
        # metric / class Metric
        # return the instance of Metric, which contains function and initial information
        #   we need for measuring the result
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        # metric / class Metric
        # return the instance of Metric, which contains function and initial information
        #   we need for measuring the result
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy', 'MAP']})
    else:
        raise ValueError('Not valid target mode')
    
    # Handle resuming the training situation
    # if cfg['resume_mode'] == 1:
    #     result = resume(cfg['model_tag'])
    #     last_epoch = result['epoch']
    #     if last_epoch > 1:
    #         model.load_state_dict(result['model_state_dict'])
    #         if cfg['model_name'] != 'base':
    #             optimizer.load_state_dict(result['optimizer_state_dict'])
    #             scheduler.load_state_dict(result['scheduler_state_dict'])
    #         logger = result['logger']
    #     else:
    #         logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    # else:
    last_epoch = 1
    logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    
    model = eval('models.{}(encoder_num_users=10, encoder_num_items=10,' 
                'decoder_num_users=10, decoder_num_items=10)'.format(cfg['model_name']))
    global_optimizer = make_optimizer(model, cfg['model_name'])
    a = global_optimizer.state_dict()
    b = a['param_groups'][0]['lr']
    global_scheduler = make_scheduler(global_optimizer, cfg['model_name'])

    # Train and Test the model for cfg[cfg['model_name']]['num_epochs'] rounds
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        logger.safe(True)
        
        global_optimizer_lr = global_optimizer.state_dict()['param_groups'][0]['lr']
        node_idx, local_parameters = train(dataset['train'], data_split['train'], data_split_info, federation, metric, logger, epoch, global_optimizer_lr)
        info = test(dataset['test'], data_split['test'], data_split_info, federation, metric, logger, epoch)

        # info = test_batchnorm(dataset['test'], data_split['test'], data_split_info, federation, metric, logger, epoch)
        model_state_dict = federation.get_new_global_model_parameter_dict()
        federation.update_global_model_momentum()
        # del local_parameters
        # gc.collect()
        global_scheduler.step()
        logger.safe(False)

        
        # if cfg['model_name'] != 'base':
        #     optimizer_state_dict = optimizer.state_dict()
        #     scheduler_state_dict = scheduler.state_dict()
        #     result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
        #               'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict,
        #               'logger': logger}
        # else:
        #     result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict, 'logger': logger}
        # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
        #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
        #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
        #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
        
        # result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict, 'logger': logger}
        
        result = {'cfg': cfg, 'epoch': epoch + 1, 'info': info, 'logger': logger, 'model_state_dict': model_state_dict}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        
        logger.reset()
    logger.safe(False)
    return


def train(dataset, data_split, data_split_info, federation, metric, logger, epoch, global_optimizer_lr):

    """
    train the model

    Parameters:
        data_loader - Object. Instance of DataLoader(data.py / make_data_loader(dataset)). 
            It constains the processed data for training. data_loader['train'] is the instance of DataLoader (class in PyTorch), 
            which is iterable (可迭代对象)
        model - Object. Instance of class AE (in models / ae.py). 
            It contains the training process of Encoder and Decoder.
        optimizer - Object. Instance of class Optimizer, which is in Pytorch(utils.py / make_optimizer()). 
            It contains the method to adjust learning rate.
        metric - Object. Instance of class Metric (metric / class Metric).
            It contains function and initial information we need for measuring the result
        logger - Object. Instance of logger.py / class Logger.
        epoch - Integer. The epoch number in for loop.

    Returns:
        None

    Raises:
        None
    """

    local, node_idx = make_local(dataset, data_split, data_split_info, federation, metric, global_optimizer_lr)
   
    num_active_nodes = len(node_idx)
    print('num_active_nodes', num_active_nodes)
    local_parameters = {}
    start_time = time.time()
    for m in range(num_active_nodes):
        # local_parameters[m] = copy.deepcopy(local[m].train(logger))
        federation.generate_new_global_model_parameter_dict(local[m].train(logger, federation, node_idx[m]), num_active_nodes)
        # cur_node_index = node_idx[m]
        # cur_local_model_dict = federation.load_local_model_dict(cur_node_index)
        # local_parameters.append(copy.deepcopy(cur_local_model_dict['model'].state_dict()))

        if m % int((num_active_nodes * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_nodes - m - 1))
            # exp_finished_time = epoch_finished_time + datetime.timedelta(
            #     seconds=round((cfg['num_epochs']['global'] - epoch) * local_time * num_active_nodes))
            exp_finished_time = 1
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_nodes),
                             'ID: {}({}/{})'.format(node_idx[m], m + 1, num_active_nodes),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
            
    # federation.combine(node_idx, local_parameters)
    return node_idx, local_parameters

def test(dataset, data_split, data_split_info, federation, metric, logger, epoch):
    if cfg['target_mode'] == 'explicit':
        metric_key = 'RMSE'
    elif cfg['target_mode'] == 'implicit':
        metric_key = 'MAP'

    with torch.no_grad():
        cur_num_users = data_split_info[0]['num_users']
        cur_num_items = data_split_info[0]['num_items']
        model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
            'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items).to(cfg["device"])'.format(cfg['model_name']))
        model = federation.distribute(model)
        # model.to(cfg['device'])
        model.train(False)

        for m in range(len(data_split)):

            cur_num_users = data_split_info[m]['num_users']
            batch_size = {'test': min(cur_num_users, cfg[cfg['model_name']]['batch_size']['test'])}
            # print('batch_size', batch_size)
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])}, batch_size)['test']
          
            for i, original_input in enumerate(data_loader):
                input = copy.deepcopy(original_input)
                input = collate(input)
                input_size = len(input)
                input = to_device(input, cfg['device'])
                output = model(input)
                # print('input', input, output)
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                # print('evaluation', evaluation)
                # if np.isnan(evaluation['RMSE']):
                #     print('input', input)
                #     print('output', output)
                if not np.isnan(evaluation[metric_key]):
                    logger.append(evaluation, 'test', input_size)
            # model.to(cfg['cpu'])
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        info = logger.write('test', metric.metric_name['test'])
        print(info)

    return info


def make_local(dataset, data_split, data_split_info, federation, metric, global_optimizer_lr):
    num_active_nodes = int(np.ceil(cfg[cfg['model_name']]['fraction'] * cfg['num_nodes']))
    # print('num_active_nodes', num_active_nodes)
    node_idx = torch.arange(cfg['num_nodes'])[torch.randperm(cfg['num_nodes'])[:num_active_nodes]].tolist()
    # local_parameters, param_idx = federation.distribute(node_idx)
    local = [None for _ in range(num_active_nodes)]
    # participated_user = []
    for m in range(num_active_nodes):
        # model_rate_m = federation.model_rate[node_idx[m]]
        cur_node_index = node_idx[m]
        user_per_node_i = data_split_info[cur_node_index]['num_users']
        # participated_user.append(user_per_node_i)
        batch_size = {'train': min(user_per_node_i, cfg[cfg['model_name']]['batch_size']['train'])}
        data_loader_m = make_data_loader({'train': SplitDataset(dataset, 
            data_split[cur_node_index])}, batch_size)['train']

        cur_num_users = data_split_info[cur_node_index]['num_users']
        cur_num_items = data_split_info[cur_node_index]['num_items']
 
        model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items)'.format(cfg['model_name']))
        model = federation.distribute(model)

        # local_optimizer = None
        # local_scheduler = None
        # if cfg['optimizer_mode'] == 'local':
        #     local_optimizer_state_dict = federation.get_local_optimizer_state_dict(cur_node_index, model)
        #     local_scheduler_state_dict = federation.get_local_scheduler_state_dict(cur_node_index, model)

        # cur_model_dict = federation.load_local_model_dict(cur_node_index)
        # federation.update_client_parameters_with_global_parameters(cur_model_dict)
        local[m] = Local(data_loader_m, model, metric, global_optimizer_lr)
    return local, node_idx


class Local:
    def __init__(self, data_loader, model, metric, global_optimizer_lr):
        self.data_loader = data_loader
        self.model = model
        self.metric = metric
        self.global_optimizer_lr = global_optimizer_lr
        # self.local_optimizer_state_dict = local_optimizer_state_dict
        # self.local_scheduler_state_dict = local_scheduler_state_dict

    def train(self, logger, federation, cur_node_index):

        model = self.model
        model.to(cfg['device'])
        model.train(True)
        
        optimizer = None
        scheduler = None

        optimizer = make_optimizer(model, cfg['model_name'])
        if cfg['optimizer_mode'] == 'local':
            # optimizer = make_optimizer(model, cfg['model_name'])
            # scheduler = make_scheduler(optimizer, cfg['model_name'])
            local_optimizer_state_dict = federation.get_local_optimizer(cur_node_index, model).state_dict()
            # local_scheduler_state_dict = federation.get_local_scheduler(cur_node_index, model).state_dict()
            optimizer.load_state_dict(local_optimizer_state_dict)
            # scheduler.load_state_dict(local_scheduler_state_dict)
        # elif cfg['optimizer_mode'] == 'global':
        #     optimizer = make_optimizer(model, cfg['model_name'])
        optimizer.param_groups[0]['lr'] = self.global_optimizer_lr
        # print('gggg', self.global_optimizer_lr, optimizer.state_dict())
        model_name = cfg['model_name']
        
        for local_epoch in range(1, cfg[model_name]['local_epoch'] + 1):
            # print('local', local)
            for i, original_input in enumerate(self.data_loader):
                input = copy.deepcopy(original_input)
                input = collate(input)
                input_size = len(input)
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output = model(input)
                
                if optimizer is not None:
                    # Zero the gradient
                    optimizer.zero_grad()
                    # Calculate the gradient of each parameter
                    output['loss'].backward()
                    # Clips gradient norm of an iterable of parameters.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # Perform a step of parameter through gradient descent Update
                    optimizer.step()
              
                evaluation = self.metric.evaluate(self.metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        if scheduler is not None:
            scheduler.step()
        if cfg['optimizer_mode'] == 'local':
            federation.store_local_optimizer(cur_node_index, copy.deepcopy(optimizer))
            # federation.store_local_scheduler(cur_node_index, copy.deepcopy(scheduler))
        # model.to('cpu')
        local_parameters = model.state_dict()
        
        return local_parameters


if __name__ == "__main__":
    main()
