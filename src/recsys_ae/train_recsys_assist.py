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
    data_split = split_dataset(dataset)
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        if 'data_split' in result:
            data_split = result['data_split']
    dataset = make_split_dataset(data_split)
    assist = Assist(data_split)
    organization = assist.make_organization()
    if cfg['target_mode'] == 'explicit':
        metric = Metric({'train': ['Loss', 'RMSE'], 'test': ['Loss', 'RMSE']})
    elif cfg['target_mode'] == 'implicit':
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    else:
        raise ValueError('Not valid target mode')
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            assist = result['assist']
            organization = result['organization']
            logger = result['logger']
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    if last_epoch == 1:
        initialize(dataset, assist, organization, metric, logger, 0)
        test(assist, metric, logger, 0)
        logger.reset()
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        dataset = assist.make_dataset(dataset, epoch)
        train(dataset, organization, metric, logger, epoch)
        organization_outputs = gather(dataset, organization, epoch)
        assist.update(organization_outputs, epoch)
        test(assist, metric, logger, epoch)
        result = {'cfg': cfg, 'epoch': epoch + 1, 'data_split': data_split, 'assist': assist,
                  'organization': organization, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return


def initialize(dataset, assist, organization, metric, logger, epoch):
    output_data = {'train': [], 'test': []}
    output_row = {'train': [], 'test': []}
    output_col = {'train': [], 'test': []}
    target_data = {'train': [], 'test': []}
    target_row = {'train': [], 'test': []}
    target_col = {'train': [], 'test': []}
    for i in range(len(dataset)):
        output_i, target_i = organization[i].initialize(dataset[i], metric, logger, epoch)
        if i % int((len(dataset) * cfg['log_interval']) + 1) == 0:
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Train Epoch: {}({:.0f}%)'.format(epoch, 100.),
                             'ID: {}/{}'.format(i + 1, len(dataset))]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
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
    logger.safe(False)
    logger.reset()
    return


def train(dataset, organization, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    num_organizations = len(organization)
    for i in range(num_organizations):
        organization[i].train(dataset[i]['train'], metric, logger, epoch)
        if i % int((num_organizations * cfg['log_interval']) + 1) == 0:
            local_time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=local_time * (num_organizations - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * local_time * num_organizations))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / num_organizations),
                             'ID: {}/{}'.format(i + 1, num_organizations),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', [metric.metric_name['train'][0]]))
    logger.safe(False)
    return


def gather(dataset, organization, epoch):
    with torch.no_grad():
        organization_outputs = [{k: None for k in dataset[i]} for i in range(len(dataset))]
        for i in range(len(dataset)):
            for k in organization_outputs[i]:
                organization_outputs[i][k] = organization[i].predict(dataset[i][k], epoch)
    return organization_outputs


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
