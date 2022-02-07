import math
import torch
import numpy as np
import torch.nn.functional as F
from config import cfg
from utils import recur


def RMSE(output, target):

    """
    RMSE between output from decoder and the target

    Parameters:
        output - torch. 
        target - torch.

    Returns:
        rmse - Integer. The rmse result

    Raises:
        None
    """

    with torch.no_grad():
        rmse = F.mse_loss(output, target).sqrt().item()
    
    # print('rmse', rmse, type(rmse))
    # if np.isnan(rmse):
    #     print('66666')
    #     print(output)
    #     print(target)
    # if not isinstance(rmse, int):
    #     print('zz', output, target)
    return rmse


def Accuracy(output, target):
    with torch.no_grad():
        batch_size = output.size(0)
        p = output.sigmoid()
        pred = torch.stack([1 - p, p], dim=-1)
        pred = pred.topk(1, 1, True, True)[1]
        correct = pred.eq(target.long().view(-1, 1).expand_as(pred)).float().sum()
        acc = (correct * (100.0 / batch_size)).item()
    return acc


def MAP(output, target, user, item, topk=10):
    user, user_idx = torch.unique(user, return_inverse=True)
    item, item_idx = torch.unique(item, return_inverse=True)
    num_users, num_items = len(user), len(item)
    if cfg['data_mode'] == 'user':
        output_ = torch.full((num_users, num_items), -float('inf'), device=output.device)
        target_ = torch.full((num_users, num_items), 0., device=target.device)
        output_[user_idx, item_idx] = output
        target_[user_idx, item_idx] = target
    elif cfg['data_mode'] == 'item':
        output_ = torch.full((num_items, num_users), -float('inf'), device=output.device)
        target_ = torch.full((num_items, num_users), 0., device=target.device)
        output_[item_idx, user_idx] = output
        target_[item_idx, user_idx] = target
    else:
        raise ValueError('Not valid data mode')
    topk = min(topk, target_.size(-1))
    idx = torch.sort(output_, dim=-1, descending=True)[1]
    topk_idx = idx[:, :topk]
    topk_target = target_[torch.arange(target_.size(0), device=output.device).view(-1, 1), topk_idx]
    precision = torch.cumsum(topk_target, dim=-1) / torch.arange(1, topk + 1, device=output.device).float()
    m = torch.sum(topk_target, dim=-1)
    ap = (precision * topk_target).sum(dim=-1) / (m + 1e-10)
    map = ap.mean().item()
    return map


class Metric(object):

    """
    Initialize metric for measuring the result.

    Parameters:
        metric_name - dict. 

    Returns:
        Instance of class Metric.

    Raises:
        None
    """

    def __init__(self, metric_name):
        # assign metric_name to self.metric_name
        self.metric_name = self.make_metric_name(metric_name)
        # assign the initial information accoring to cfg['data_name'] and cfg['target_mode']
        self.pivot, self.pivot_name, self.pivot_direction = self.make_pivot()
        # assign the lambda function to the metric (dict)
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'RMSE': (lambda input, output: RMSE(output['target_rating'], input['target_rating'])),
                       'Accuracy': (lambda input, output: Accuracy(output['target_rating'], input['target_rating'])),
                       'MAP': (lambda input, output: MAP(output['target_rating'], input['target_rating'],
                                                         input['target_user'], input['target_item']))}

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']:
            if cfg['target_mode'] == 'explicit':
                pivot = float('inf')
                pivot_direction = 'down'
                pivot_name = 'RMSE'
            elif cfg['target_mode'] == 'implicit':
                pivot = -float('inf')
                pivot_direction = 'up'
                pivot_name = 'MAP'
            else:
                raise ValueError('Not valid target mode')
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation

    def compare(self, val):
        if self.pivot_direction == 'down':
            compared = self.pivot > val
        elif self.pivot_direction == 'up':
            compared = self.pivot < val
        else:
            raise ValueError('Not valid pivot direction')
        return compared

    def update(self, val):
        self.pivot = val
        return
