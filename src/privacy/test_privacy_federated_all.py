import argparse
import os
import copy
import torch
import torch.backends.cudnn as cudnn
import models
import numpy as np
import collections

from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, SplitDataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, resume, collate, make_optimizer, make_scheduler
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
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    dataset = fetch_dataset(cfg['data_name'])
    process_dataset(dataset)
    data_split, data_split_info = split_dataset(dataset, cfg['num_nodes'], cfg['data_split_mode'])
    # data_loader = make_data_loader(dataset)
    
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'MAP'], 'test': ['Loss', 'Accuracy', 'MAP']})
    else:
        raise ValueError('Not valid target mode')
    
    result = resume(cfg['model_tag'], load_tag='best')
    # print(result['info'])
    global_model_state_dict = result['model_state_dict']
    last_epoch = result['epoch']
    model.load_state_dict(global_model_state_dict)
    test_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    test(dataset['test'], data_split['test'], data_split_info, model, metric, test_logger, last_epoch)

    if cfg['fine_tune'] == True:
        train_logger = result['logger'] if 'logger' in result else None
        fine_tune(dataset, data_split, data_split_info, global_model_state_dict, metric, train_logger, test_logger)
    
    # result = resume(cfg['model_tag'], load_tag='checkpoint')
    # train_logger = result['logger'] if 'logger' in result else None
    # result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    # save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return

def test(dataset, data_split, data_split_info, model, metric, logger, epoch):
    if cfg['target_mode'] == 'explicit':
        metric_key = {'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']}
    elif cfg['target_mode'] == 'implicit':
        metric_key = {'train': ['Loss', 'MAP'], 'test': ['Loss', 'Accuracy', 'MAP']}

    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for m in range(len(data_split_info)):
            cur_num_users = data_split_info[m]['num_users']
            batch_size = {'test': min(cur_num_users, cfg[cfg['model_name']]['batch_size']['test'])}
            # print('batch_size', batch_size)
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])}, batch_size)['test']
          
            for i, original_input in enumerate(data_loader):
                input = copy.deepcopy(original_input)
                input = collate(input)
                input_size = len(input['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output = model(input)
                # print('input', input, output)
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                # print('evaluation', evaluation)
                # if np.isnan(evaluation['RMSE']):
                #     print('input', input)
                #     print('output', output)
                if not np.isnan(evaluation[metric_key['test'][-1]]):
                    logger.append(evaluation, 'test', input_size)
            # model.to(cfg['cpu'])
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        info = logger.write('test', metric.metric_name['test'])
        print(info)
    logger.safe(False)
    return

def fine_tune(dataset, data_split, data_split_info, global_model_state_dict, metric, train_logger, test_logger):
    if cfg['target_mode'] == 'explicit':
        metric_key = {'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']}
    elif cfg['target_mode'] == 'implicit':
        metric_key = {'train': ['Loss', 'MAP'], 'test': ['Loss', 'Accuracy', 'MAP']}

    train_dataset = dataset['train']
    train_data_split = data_split['train']
    test_dataset = dataset['test']
    test_data_split = data_split['test']
    train_evaluation_record = collections.defaultdict(dict)
    test_evaluation_record = collections.defaultdict(dict)
    # Iterate through all nodes
    for m in range(len(data_split_info)):
        if m % int((len(data_split_info) * cfg['log_interval']) + 1) == 0:
            print('cur_fine_tune_percentage', m, len(data_split_info), str(100 * m / len(data_split_info))+'%')
        cur_num_users = data_split_info[m]['num_users']
        cur_num_items = data_split_info[m]['num_items']
        batch_size = {'train': min(cur_num_users, cfg['fine_tune_batch_size'])}
        train_data_loader_m = make_data_loader({'train': SplitDataset(train_dataset, 
                train_data_split[m])}, batch_size)['train']
        batch_size = {'test': min(cur_num_users, cfg['fine_tune_batch_size'])}
        test_data_loader_m = make_data_loader({'test': SplitDataset(test_dataset, 
                test_data_split[m])}, batch_size)['test']

        model = eval('models.{}(encoder_num_users=cur_num_users, encoder_num_items=cur_num_items,' 
                'decoder_num_users=cur_num_users, decoder_num_items=cur_num_items)'.format(cfg['model_name']))
        model.load_state_dict(copy.deepcopy(global_model_state_dict))

        optimizer = make_optimizer(model, cfg['model_name'], is_fine_tune=True)
        scheduler = make_scheduler(optimizer, cfg['model_name'], is_fine_tune=True)

        for epoch in range(1, cfg['fine_tune_epoch']+1):
            train_total_input_size = 0
            train_total_evaluation = collections.defaultdict(int)
            for i, original_input in enumerate(train_data_loader_m):
                input = copy.deepcopy(original_input)
                input = collate(input)
                input_size = len(input)
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
                evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                # print('evaluation', evaluation)
                if not np.isnan(evaluation[metric_key['train'][-1]]):
                    train_total_input_size += input_size
                    for key in evaluation:
                        train_total_evaluation[key] += evaluation[key] * input_size
            if scheduler is not None:
                scheduler.step()
            # print('train_total_evaluation', train_total_evaluation)
            train_evaluation_record[m][epoch] = collections.defaultdict(dict)
            train_evaluation_record[m][epoch]['train_total_input_size'] = train_total_input_size
            train_evaluation_record[m][epoch]['train_total_evaluation'] = copy.deepcopy(train_total_evaluation)

            test_total_input_size = 0
            test_total_evaluation = collections.defaultdict(int)
            for i, original_input in enumerate(test_data_loader_m):
                input = copy.deepcopy(original_input)
                input = collate(input)
                input_size = len(input)
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                if not np.isnan(evaluation[metric_key['test'][-1]]):
                    test_total_input_size += input_size
                    for key in evaluation:
                        test_total_evaluation[key] += evaluation[key] * input_size
            test_evaluation_record[m][epoch] = collections.defaultdict(dict)
            test_evaluation_record[m][epoch]['test_total_input_size'] = test_total_input_size
            test_evaluation_record[m][epoch]['test_total_evaluation'] = copy.deepcopy(test_total_evaluation)

    for epoch in range(1, cfg['fine_tune_epoch']+1):
        train_logger.safe(True)
        for m in train_evaluation_record.keys():
            evaluation = train_evaluation_record[m][epoch]['train_total_evaluation']
            input_size = train_evaluation_record[m][epoch]['train_total_input_size']
            train_logger.append(evaluation, 'train', input_size, mean=True, is_fine_tune=True)
            if m % int((len(data_split_info) * cfg['log_interval']) + 1) == 0:
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                             'fine_tune_Train_Epoch: {}({:.0f}%)'.format(epoch, 100. * m / len(data_split_info)),
                             'ID: {}({}/{})'.format(m, m + 1, len(data_split_info))]}
                train_logger.append(info, 'train', mean=False)
                print(train_logger.write('train', metric.metric_name['train']))
        train_logger.safe(False)

        test_logger.safe(True)
        for m in test_evaluation_record.keys():
            evaluation = test_evaluation_record[m][epoch]['test_total_evaluation']
            input_size = test_evaluation_record[m][epoch]['test_total_input_size']
            test_logger.append(evaluation, 'test', input_size, mean=True, is_fine_tune=True)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 
                         'fine_tune_Test_Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        test_logger.append(info, 'test', mean=False)
        # print(test_logger.write('test', metric.metric_name['test']))
        test_logger.safe(False)
    # print('train_evaluation_record', train_evaluation_record)
    # print('test_evaluation_record', test_evaluation_record)
    return

if __name__ == "__main__":
    main()
