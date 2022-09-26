import argparse
import copy
import datetime
from platform import node

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
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])

        runExperiment()
    return

def runExperiment():

    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)

    data_split, data_split_info = split_dataset(dataset, cfg['num_nodes'], cfg['data_split_mode'])

    federation = Federation(data_split_info)
    federation.create_local_optimizer()

    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'NDCG'], 'test': ['Loss', 'NDCG']})
    else:
        raise ValueError('Not valid target mode')
    
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    global_optimizer = make_optimizer(model, cfg['model_name'], 'client')
    global_scheduler = make_scheduler(global_optimizer, cfg['model_name'], 'client')

    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            federation.server_model.load_state_dict(result['model_state_dict'])
            if cfg['model_name'] != 'base':
                global_optimizer.load_state_dict(result['global_optimizer_state_dict'])
                global_scheduler.load_state_dict(result['global_scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger('./output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('./output/runs/train_{}'.format(cfg['model_tag']))
    
    for epoch in range(last_epoch, cfg['client'][cfg['model_name']]['num_epochs'] + 1):
        logger.safe(True)
        
        global_optimizer_lr = global_optimizer.state_dict()['param_groups'][0]['lr']
        node_idx = train(
            dataset['train'], 
            data_split['train'], 
            data_split_info, 
            federation, 
            metric, 
            logger, 
            epoch, 
            global_optimizer_lr
        )
        federation.update_server_model_momentum()
        model_state_dict = federation.server_model.state_dict()
        info = test(
            dataset['test'], 
            data_split['test'], 
            data_split_info, 
            federation, 
            metric, 
            logger, 
            epoch
        )
    
        global_scheduler.step()
        logger.safe(False)

        global_optimizer_state_dict = global_optimizer.state_dict()
        global_scheduler_state_dict = global_scheduler.state_dict()

        result = {
            'cfg': cfg, 
            'epoch': epoch + 1, 
            'info': info, 
            'logger': logger, 
            'model_state_dict': model_state_dict, 
            'global_optimizer_state_dict': global_optimizer_state_dict, 
            'global_scheduler_state_dict': global_scheduler_state_dict,
            'data_split': data_split, 
            'data_split_info': data_split_info
        }

        checkpoint_path = './output/model/{}_checkpoint.pt'.format(cfg['model_tag'])
        best_path = './output/model/{}_best.pt'.format(cfg['model_tag'])
        save(result, checkpoint_path)
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy(checkpoint_path, best_path)
        
        logger.reset()
    logger.safe(False)
    return

def train(
    dataset, 
    data_split, 
    data_split_info, 
    federation, 
    metric, 
    logger, 
    epoch, 
    global_optimizer_lr
):

    local, node_idx = make_local(
        dataset, 
        data_split, 
        data_split_info, 
        federation, 
        metric
    )
   
    num_active_nodes = len(node_idx)

    start_time = time.time()
    for m in range(num_active_nodes):    
        local_parameters = local[m].train(logger, federation, node_idx[m], global_optimizer_lr)
        federation.generate_new_server_model_parameter_dict(
            model_state_dict=local_parameters, 
            total_client=num_active_nodes
        )
       
        if m % int((num_active_nodes * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (m + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_active_nodes - m - 1))
            model_name = cfg['model_name']
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['client'][model_name]['num_epochs'] - epoch) * local_time * len(node_idx)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * m / num_active_nodes),
                             'ID: {}({}/{})'.format(node_idx[m], m + 1, num_active_nodes),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
            
    return node_idx

def test(
    dataset, 
    data_split, 
    data_split_info, 
    federation, 
    metric, 
    logger, 
    epoch
):

    with torch.no_grad():
        cur_num_users = data_split_info[0]['num_users']
        cur_num_items = data_split_info[0]['num_items']
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model = federation.update_client_parameters_with_server_model_parameters(model)
        model.to(cfg['device'])
        model.train(False)

        for m in range(len(data_split)):

            cur_num_users = data_split_info[m]['num_users']
            batch_size = {
                'test': min(cur_num_users, cfg['client'][cfg['model_name']]['batch_size']['test'])
            }
            data_loader = make_data_loader(
                {'test': SplitDataset(dataset, data_split[m])}, 
                batch_size
            )['test']
          
            for i, input in enumerate(data_loader):
                input = collate(input)
                input_size = len(input['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output = model(input)
                if cfg['experiment_size'] == 'large':
                    input = to_device(input, 'cpu')
                    output = to_device(output, 'cpu')

                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
        
        if cfg['experiment_size'] == 'large':
            model.to('cpu')

        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        info = logger.write('test', metric.metric_name['test'])
        print(info)

    return info


def make_local(dataset, data_split, data_split_info, federation, metric):
    num_active_nodes = int(np.ceil(cfg['client'][cfg['model_name']]['fraction'] * cfg['num_nodes']))
    node_idx = torch.arange(cfg['num_nodes'])[torch.randperm(cfg['num_nodes'])[:num_active_nodes]].tolist()
    local = [None for _ in range(num_active_nodes)]

    for m in range(num_active_nodes):
        cur_node_index = node_idx[m]
        user_per_node_i = data_split_info[cur_node_index]['num_users']

        batch_size = {
            'train': min(user_per_node_i, cfg['client'][cfg['model_name']]['batch_size']['train'])
        }

        data_loader_m = make_data_loader(
            {'train': SplitDataset(dataset, data_split[cur_node_index])}, 
            batch_size
        )['train']
 
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model = federation.update_client_parameters_with_server_model_parameters(model)
        local[m] = Local(data_loader_m, model, metric)
    return local, node_idx


class Local:
    def __init__(self, data_loader, model, metric):
        self.data_loader = data_loader
        self.model = model
        self.metric = metric

    def train(self, logger, federation, cur_node_index, global_optimizer_lr):

        model = self.model
        model.to(cfg['device'])
        model.train(True)
        
        optimizer = None
        scheduler = None

        optimizer = make_optimizer(model, cfg['model_name'], 'client')      
        local_optimizer_state_dict = federation.get_local_optimizer_state_dict(cur_node_index) 
        local_optimizer_state_dict = to_device(local_optimizer_state_dict, cfg['device'])
        optimizer.load_state_dict(local_optimizer_state_dict) 
        optimizer.param_groups[0]['lr'] = global_optimizer_lr
        for local_epoch in range(1, cfg['client'][cfg['model_name']]['local_epoch'] + 1):
            for i, input in enumerate(self.data_loader):

                input = collate(input)
                input_size = len(input['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output = model(input)
                if optimizer is not None:
                    optimizer.zero_grad()
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()

                if cfg['experiment_size'] == 'large':
                    input = to_device(input, 'cpu')
                    output = to_device(output, 'cpu')

                evaluation = self.metric.evaluate(self.metric.metric_name['train'], input, output)
                logger.append(evaluation, 'train', n=input_size)
        
        optimizer_state_dict = optimizer.state_dict()
        if cfg['experiment_size'] == 'large':
            model.to('cpu')
            optimizer_state_dict = to_device(optimizer_state_dict, 'cpu')

        federation.store_local_optimizer_state_dict(cur_node_index, copy.deepcopy(optimizer_state_dict)) 
        local_parameters = model.state_dict()
        
        return local_parameters


if __name__ == "__main__":
    main()
