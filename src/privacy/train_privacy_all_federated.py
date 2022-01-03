import argparse
import copy
import datetime
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
import collections
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
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
    dataset = fetch_dataset(cfg['data_name'])

    # utils.py / process_dataset(dataset)
    # add some key:value (size, num) to cfg
    process_dataset(dataset)
    
    # data.py / make_data_loader(dataset)
    # data_loader is a dict, has 2 keys - 'train', 'test'
    # data_loader['train'] is the instance of DataLoader (class in PyTorch), which is iterable (可迭代对象)
    data_loader = make_data_loader(dataset)

    # models / cfg['model_name'].py initializes the model, for example, models / ae.py / class AE
    # .to(cfg["device"]) means copy the tensor to the specific GPU or CPU, and run the 
    # calculation there.
    # model is the instance of class AE (in models / ae.py). It contains the training process of 
    #   Encoder and Decoder.

    
    # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    # a = sys.getsizeof(model)

    # print("model", model)
    model_dict = collections.defaultdict(list)
    # model_list = []

    # generate model for all unique user
    for i in range(cfg['unique_user_num']):

        cur_name = 'model_' + str(i)
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model_dict[cur_name].append(model)
        
        # root = processed_folder(i, True)
        # model_path = os.path.join(root, cfg['model_name'] + '.pt')
        # save(model, model_path)

        if cfg['model_name'] != 'base':
            # utils.py / make_optimizer()
            optimizer = make_optimizer(model, cfg['model_name'])
            # utils.py / make_scheduler()
            scheduler = make_scheduler(optimizer, cfg['model_name'])
        else:
            optimizer = None
            scheduler = None
        # print(i, id(model), id(optimizer), id(scheduler))
        model_dict[cur_name].append(optimizer)
        model_dict[cur_name].append(scheduler)
        # model_list.append([model, optimizer, scheduler])


        # optimizer_path = os.path.join(root, cfg['model_name'] + '_optimizer' + '.pt')
        # scheduler_path = os.path.join(root, cfg['model_name'] + '_scheduler' + '.pt')
        # save(optimizer, optimizer_path)
        # save(scheduler, scheduler_path)

        # del model
        # del optimizer
        # del scheduler
        # gc.collect()


    if cfg['target_mode'] == 'explicit':
        # metric / class Metric
        # return the instance of Metric, which contains function and initial information
        #   we need for measuring the result
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        # metric / class Metric
        # return the instance of Metric, which contains function and initial information
        #   we need for measuring the result
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
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

    # Use multiple GPU to accelarate training
    # if cfg['world_size'] > 1:
    #     model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))

    # Train and Test the model for cfg[cfg['model_name']]['num_epochs'] rounds
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):

        # root = processed_folder(i, True)
        # model_path = os.path.join(root, cfg['model_name'] + '.pt')
        # model = load(model_path)
        # model = model.to(cfg["device"])
        # optimizer_path = os.path.join(root, cfg['model_name'] + '_optimizer' + '.pt')
        # optimizer = load(optimizer_path)
        # scheduler_path = os.path.join(root, cfg['model_name'] + '_scheduler' + '.pt')
        # scheduler = load(scheduler_path)

        # cur_model_list = model_list[i]
        # model = cur_model_list[0]
        # optimizer = cur_model_list[1]
        # scheduler = cur_model_list[2]
        
        # data_loader['train'][i]
        # train(data_loader['train'], model, optimizer, metric, logger, epoch, i)
        # test(data_loader['test'], model, metric, logger, epoch, i)
        
        train(data_loader['train'], model_dict, metric, logger, epoch, data_loader['test'])
        test(data_loader['test'], model_dict, metric, logger, epoch)
        
        if scheduler is not None:
            for key in model_dict:
                model_dict[key][2].step()

        # model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
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
        logger.reset()
    
    for key in model_dict:
        a = model_dict[key][0].state_dict()
    return

def reset_parameters(model):
    for m in model.blocks:
        # if item m is nn.Linear, set its value to xavier_uniform heuristic value
        if isinstance(m, nn.Linear):
            m.weight.data.zero_()
            if m.bias is not None:
                # set bias to 0
                m.bias.data.zero_()


def train(data_loader, model_dict, metric, logger, epoch, test_data):

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

    logger.safe(True)
    # Set the model in training mode
    
    start_time = time.time()
    global_decoder_model = copy.deepcopy(cfg['Decoder_instance'])
    reset_parameters(global_decoder_model)
    global_encoder_model = copy.deepcopy(cfg['Encoder_instance'])
    reset_parameters(global_encoder_model)

    model_name = cfg['model_name']
    client_fraction = cfg[model_name]['fraction']
    client_count = math.ceil(cfg['unique_user_num'] * client_fraction)
    print('client_count', client_count)
    cur_client_count = 0

    model_name = cfg['model_name']
    if cfg[model_name]['fraction'] < 1:
        picked_index = random.sample(range(0, cfg['unique_user_num']), client_count)
        picked_index_dict = {}
        for index in picked_index:
            picked_index_dict[index] = 1

        # print('picked_index', picked_index)
    # Iterate data_loader
    for i, input in enumerate(data_loader):
        
        # if cur_client_count < client_count:
        #     cur_client_count += 1
        # else:
        #     break
        # print('cur_client_count', cur_client_count)
        # concatenate model_ and the picked index
        if cfg[model_name]['fraction'] < 1:
            if i not in picked_index_dict:
                # print("i", i)
                continue

        model_name = cfg['model_name']
        if cfg[model_name]['batch_size']['train'] > 1:
            cur_name = 'model_' + str(i)
        else:
            cur_name = 'model_' + str(input['user'][0][0].item())

        model = model_dict[cur_name][0]
        optimizer = model_dict[cur_name][1]
        model.train(True)

        # utils.py / collate(input)
        # input is the batch_size data that has been processed by input_collate(batch)
        # input_collate(batch) is in data.py / input_collate(batch)
        input = collate(input)
        input_size = len(input[cfg['data_mode']])
        if input_size == 0:
            continue
        input = to_device(input, cfg['device'])
        # input['epoch'] = epoch
        
        for local_epoch in range(cfg[model_name]['local_epoch']):
            temp_input = copy.deepcopy(input)
            # put the input in model => forward() => train Encoder and Decoder and get loss
            # temp_input['cur_local_epoch'] = local_epoch
            temp_input['cur_mode'] = 'train'
            output = model(temp_input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']

            # test(test_data, model_dict, metric, logger, epoch)

            # update parameters of model
            if optimizer is not None:
                # Zero the gradient
                optimizer.zero_grad()
                # Calculate the gradient of each parameter
                output['loss'].backward()
                # Clips gradient norm of an iterable of parameters.
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                # Perform a step of parameter through gradient descent Update
                optimizer.step()

            evaluation = metric.evaluate(metric.metric_name['train'], temp_input, output)
        
        logger.append(evaluation, 'train', n=input_size)
        # Record information when epoch is a multiple of a certain number
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {},{}'.format(cfg['model_tag'], 'user'+ str(i)),
                            'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                            'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                            'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))

        # named_parameters() returns back list of tuple (generator)
        for key, value in model.decoder.named_parameters():
            # state_dict() returns back dictionary
            global_decoder_model.state_dict()[key] += model.decoder.state_dict()[key]

        for key, value in model.encoder.named_parameters():
            global_encoder_model.state_dict()[key] += model.encoder.state_dict()[key]

    if cfg['train_mode'] == "private":
        for key, value in model.decoder.named_parameters():
            global_decoder_model.state_dict()[key] /= client_count
        
        if cfg['federated_mode'] == "all":
            for key, value in model.encoder.named_parameters():
                global_encoder_model.state_dict()[key] /= client_count

    save(global_decoder_model, processed_folder(epoch, False))
    print('client_count', client_count)
    cfg['global_decoder_model'] = global_decoder_model
    cfg['global_encoder_model'] = global_encoder_model
    logger.safe(False)
    return


def test(data_loader, model_dict, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad(): 
        for i, input in enumerate(data_loader):
            
            model_name = cfg['model_name']
            if cfg[model_name]['batch_size']['train'] > 1:
                cur_name = 'model_' + str(i)
                # print('cur_name', cur_name)
            else:
                cur_name = 'model_' + str(input['user'][0][0].item())
            model = model_dict[cur_name][0]
            model.train(False)

            input = collate(input)
            input_size = len(input['target_{}'.format(cfg['data_mode'])])
            if input_size == 0:
                continue
            input = to_device(input, cfg['device'])
            input['cur_mode'] = 'test'
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    logger.safe(False)
    return


if __name__ == "__main__":
    main()