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
from utils import save, to_device, process_control, process_dataset, resume, collate, make_optimizer, make_scheduler, fix_parameters, init_final_layer
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

    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'NDCG'], 'test': ['Loss', 'NDCG']})
    else:
        raise ValueError('Not valid target mode')
    
    result = resume(cfg['model_tag'], load_tag='best')
    global_model_state_dict = result['model_state_dict']
    last_epoch = result['epoch']
    data_split = result['data_split']
    data_split['test'] = copy.deepcopy(data_split['train'])
    data_split_info = result['data_split_info']
    model.load_state_dict(global_model_state_dict)
    test_logger = make_logger('./output/runs/test_{}'.format(cfg['model_tag']))
    test(dataset['test'], data_split['test'], data_split_info, model, metric, test_logger, last_epoch)

    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    
    result = {'cfg': cfg, 'epoch': last_epoch, 'logger': {'train': train_logger, 'test': test_logger}}
    save(result, './output/result/{}.pt'.format(cfg['model_tag']))
    return

def test(dataset, data_split, data_split_info, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for m in range(len(data_split_info)):
            cur_num_users = data_split_info[m]['num_users']
            batch_size = {'test': min(cur_num_users, cfg['client'][cfg['model_name']]['batch_size']['test'])}
            data_loader = make_data_loader({'test': SplitDataset(dataset, data_split[m])}, batch_size)['test']
          
            for i, original_input in enumerate(data_loader):
                input = copy.deepcopy(original_input)
                input = collate(input)
                input_size = len(input['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                input = to_device(input, cfg['device'])
                output = model(input)
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']),
                         'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        info = logger.write('test', metric.metric_name['test'])
        print(info)
    logger.safe(False)
    return

if __name__ == "__main__":
    main()
