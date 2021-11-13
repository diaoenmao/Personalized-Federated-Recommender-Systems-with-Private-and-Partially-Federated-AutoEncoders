import argparse
import copy
import datetime
import models
import os
import shutil
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, make_split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
from assist import Assist
from scipy.sparse import csr_matrix

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
    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy', 'MAP']})
    else:
        raise ValueError('Not valid target mode')
    result = resume(cfg['model_tag'])
    last_epoch = result['epoch']
    data_split = result['data_split']
    dataset = make_split_dataset(data_split)
    dataset = [{'test': dataset[i]['test']} for i in range(len(dataset))]
    assist = result['assist']
    assist.reset()
    organization = result['organization']
    test_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    test_each_logger = make_logger('output/runs/test_{}'.format(cfg['model_tag']))
    initialize(dataset, assist, organization, 0)
    test_each(assist, metric, test_each_logger, 0)
    test(assist, metric, test_logger, 0)
    test_logger.reset()
    for epoch in range(1, last_epoch):
        dataset = assist.make_dataset(dataset, epoch)
        organization_outputs = gather(dataset, organization, epoch)
        assist.update(organization_outputs, epoch)
        test_each(assist, metric, test_each_logger, epoch)
        test(assist, metric, test_logger, epoch)
        test_logger.reset()
    assist.reset()
    result = resume(cfg['model_tag'], load_tag='checkpoint')
    train_logger = result['logger'] if 'logger' in result else None
    save_result = {'cfg': cfg, 'epoch': last_epoch, 'assist': assist,
                   'logger': {'train': train_logger, 'test': test_logger, 'test_each': test_each_logger}}
    save(save_result, './output/result/{}.pt'.format(cfg['model_tag']))
    return


def initialize(dataset, assist, organization, epoch):
    output_data = {'test': []}
    output_row = {'test': []}
    output_col = {'test': []}
    target_data = {'test': []}
    target_row = {'test': []}
    target_col = {'test': []}
    for i in range(len(dataset)):
        output_i, target_i = organization[i].initialize(dataset[i], None, None, epoch)
        for k in dataset[0]:
            output_coo_i_k = output_i[k].tocoo()
            output_data[k].append(output_coo_i_k.data)
            output_row[k].append(output_coo_i_k.row)
            output_col[k].append(output_coo_i_k.col)
            target_coo_i_k = target_i[k].tocoo()
            target_data[k].append(target_coo_i_k.data)
            target_row[k].append(target_coo_i_k.row)
            target_col[k].append(target_coo_i_k.col)
    if cfg['data_mode'] == 'user':
        for k in dataset[0]:
            assist.organization_output[0][k] = csr_matrix(
                (np.concatenate(output_data[k]), (np.concatenate(output_row[k]), np.concatenate(output_col[k]))),
                shape=(cfg['num_users']['target'], cfg['num_items']['target']))
            assist.organization_target[0][k] = csr_matrix(
                (np.concatenate(target_data[k]), (np.concatenate(target_row[k]), np.concatenate(target_col[k]))),
                shape=(cfg['num_users']['target'], cfg['num_items']['target']))
    elif cfg['data_mode'] == 'item':
        for k in dataset[0]:
            assist.organization_output[0][k] = csr_matrix(
                (np.concatenate(output_data[k]), (np.concatenate(output_row[k]), np.concatenate(output_col[k]))),
                shape=(cfg['num_items']['target'], cfg['num_users']['target']))
            assist.organization_target[0][k] = csr_matrix(
                (np.concatenate(target_data[k]), (np.concatenate(target_row[k]), np.concatenate(target_col[k]))),
                shape=(cfg['num_items']['target'], cfg['num_users']['target']))
    else:
        raise ValueError('Not valid data mode')
    return


def gather(dataset, organization, epoch):
    with torch.no_grad():
        organization_outputs = [{split: None for split in dataset[i]} for i in range(len(dataset))]
        for i in range(len(dataset)):
            for split in organization_outputs[i]:
                organization_outputs[i][split] = organization[i].predict(dataset[i][split], epoch)
    return organization_outputs


def test_each(assist, metric, each_logger, epoch):
    with torch.no_grad():
        organization_output = assist.organization_output[epoch]['test']
        organization_target = assist.organization_target[0]['test']
        batch_size = cfg[cfg['model_name']]['batch_size']['test']
        for i in range(len(assist.data_split)):
            each_logger.safe(True)
            output_i = organization_output[:, assist.data_split[i]]
            target_i = organization_target[:, assist.data_split[i]]
            for j in range(0, output_i.shape[0], batch_size):
                output_i_j = output_i[j:j + batch_size]
                target_i_j = target_i[j:j + batch_size]
                output_i_j_coo = output_i_j.tocoo()
                output_i_j_rating = torch.tensor(output_i_j_coo.data)
                target_i_j_coo = target_i_j.tocoo()
                target_i_j_user = torch.tensor(target_i_j_coo.row, dtype=torch.long)
                target_i_j_item = torch.tensor(target_i_j_coo.col, dtype=torch.long)
                target_i_j_rating = torch.tensor(target_i_j_coo.data)
                output = {'target_rating': output_i_j_rating}
                input = {'target_rating': target_i_j_rating, 'target_user': target_i_j_user,
                         'target_item': target_i_j_item}
                input_size = len(input['target_{}'.format(cfg['data_mode'])])
                if input_size == 0:
                    continue
                output['loss'] = models.loss_fn(output_i_j_rating, target_i_j_rating)
                output = to_device(output, cfg['device'])
                input = to_device(input, cfg['device'])
                evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                each_logger.append(evaluation, 'test', n=input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                             'ID: {}/{}'.format(i + 1, len(assist.data_split))]}
            each_logger.append(info, 'test', mean=False)
            print(each_logger.write('test', metric.metric_name['test']))
            each_logger.safe(False)
            each_logger.reset()
    return


def test(assist, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        organization_output = assist.organization_output[epoch]['test']
        organization_target = assist.organization_target[0]['test']
        batch_size = cfg[cfg['model_name']]['batch_size']['test']
        for i in range(0, organization_output.shape[0], batch_size):
            output_i = organization_output[i:i + batch_size]
            target_i = organization_target[i:i + batch_size]
            output_i_coo = output_i.tocoo()
            output_i_rating = torch.tensor(output_i_coo.data)
            target_i_coo = target_i.tocoo()
            if cfg['data_mode'] == 'user':
                target_i_user = torch.tensor(target_i_coo.row, dtype=torch.long)
                target_i_item = torch.tensor(target_i_coo.col, dtype=torch.long)
            elif cfg['data_mode'] == 'item':
                target_i_user = torch.tensor(target_i_coo.col, dtype=torch.long)
                target_i_item = torch.tensor(target_i_coo.row, dtype=torch.long)
            else:
                raise ValueError('Not valid data mode')
            target_i_rating = torch.tensor(target_i_coo.data)
            output = {'target_rating': output_i_rating}
            input = {'target_rating': target_i_rating, 'target_user': target_i_user,
                     'target_item': target_i_item}
            input_size = len(input['target_{}'.format(cfg['data_mode'])])
            if input_size == 0:
                continue
            output['loss'] = models.loss_fn(output_i_rating, target_i_rating)
            output = to_device(output, cfg['device'])
            input = to_device(input, cfg['device'])
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', n=input_size)
        lr = assist.ar_state_dict[epoch][0]['assist_rate'].item() if assist.ar_state_dict[epoch][0] is not None else 0
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.),
                         'Assist rate: {:.6f}'.format(lr)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
        logger.safe(False)
    return


if __name__ == "__main__":
    main()
