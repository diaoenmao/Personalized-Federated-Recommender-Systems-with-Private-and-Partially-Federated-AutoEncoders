import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import copy
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
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
    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    print("model", model)

    if cfg['model_name'] != 'base':
        # utils.py / make_optimizer()
        optimizer = make_optimizer(model, cfg['model_name'])
        # utils.py / make_scheduler()
        scheduler = make_scheduler(optimizer, cfg['model_name'])
    else:
        optimizer = None
        scheduler = None
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
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'])
        last_epoch = result['epoch']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            if cfg['model_name'] != 'base':
                optimizer.load_state_dict(result['optimizer_state_dict'])
                scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))
    else:
        last_epoch = 1
        logger = make_logger('output/runs/train_{}'.format(cfg['model_tag']))

    # Use multiple GPU to accelarate training
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))

    # Train and Test the model for cfg[cfg['model_name']]['num_epochs'] rounds
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        train(data_loader['train'], model, optimizer, metric, logger, epoch)
        test(data_loader['test'], model, metric, logger, epoch)
        if scheduler is not None:
            scheduler.step()
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        if cfg['model_name'] != 'base':
            optimizer_state_dict = optimizer.state_dict()
            scheduler_state_dict = scheduler.state_dict()
            result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict,
                      'optimizer_state_dict': optimizer_state_dict, 'scheduler_state_dict': scheduler_state_dict,
                      'logger': logger}
        else:
            result = {'cfg': cfg, 'epoch': epoch + 1, 'model_state_dict': model_state_dict, 'logger': logger}
        save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
        if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                        './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    return

def processed_folder(epoch):
    root = './federated_decoder/'
    root = os.path.expanduser(root)
    res = os.path.join(root, str(cfg['private_decoder_user']), str(epoch), 'federated_decoder.pt')
    return res

def train(data_loader, model, optimizer, metric, logger, epoch):

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
    model.train(True)
    start_time = time.time()
    server_decoder_model = copy.deepcopy(cfg['Decoder_instance'])
    # Iterate data_loader
    for i, input in enumerate(data_loader):
        # utils.py / collate(input)
        # input is the batch_size data that has been processed by input_collate(batch)
        # input_collate(batch) is in data.py / input_collate(batch)
        input = collate(input)
        input_size = len(input[cfg['data_mode']])
        if input_size == 0:
            continue
        input = to_device(input, cfg['device'])
        input['epoch'] = epoch
        # put the input in model => forward() => train Encoder and Decoder and get loss
        output = model(input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']

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

        # named_parameters() returns back list of tuple (generator)
        for key, value in model.decoder.named_parameters():
            # state_dict() returns back dictionary
            server_decoder_model.state_dict()[key] += model.decoder.state_dict()[key]

        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)

        # Record information when epoch is a multiple of a certain number
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr'] if optimizer is not None else 0
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))

    if cfg['train_mode'] == "private":
        for key, value in model.decoder.named_parameters():
            server_decoder_model.state_dict()[key] = torch.div(server_decoder_model.state_dict()[key], cfg['private_decoder_user'])

    save(server_decoder_model, processed_folder(epoch), mode='pickle')
    
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = len(input['target_{}'.format(cfg['data_mode'])])
            if input_size == 0:
                continue
            input = to_device(input, cfg['device'])
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
    print("5555")
    main()
