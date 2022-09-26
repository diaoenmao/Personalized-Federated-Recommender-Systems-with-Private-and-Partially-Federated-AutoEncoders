import math
import copy
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

def div_no_nan(a, b, na_value=0.):
    return (a / b).nan_to_num_(nan=na_value, posinf=na_value, neginf=na_value)


def DCG(target):
    batch_size, k = target.shape
    rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
    dcg = (target / torch.log2(rank_positions + 1)).sum(dim=-1)
    return dcg


def NDCG(output, target, user, item, topk=10):
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
    topk_ = min(topk, output_.size(-1))
    _, output_topk_idx = output_.topk(topk_, dim=-1)
    sorted_target = target_.take_along_dim(output_topk_idx, dim=-1)
    ideal_target, _ = target_.topk(topk_, dim=-1)
    ndcg = div_no_nan(DCG(sorted_target), DCG(ideal_target)).mean().item()
    return ndcg


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
        self.metric = {
            'Loss': (lambda input, output: output['loss'].item()),
            'RMSE': (lambda input, output: RMSE(output['target_rating'], input['target_rating'])),
            'Accuracy': (lambda input, output: Accuracy(output['target_rating'], input['target_rating'])),
            'MAP': (lambda input, output: MAP(output['target_rating'], input['target_rating'],
                                                input['target_user'], input['target_item'])),
            'NDCG': (lambda input, output: NDCG(output['target_rating'], input['target_rating'],
                                                           input['target_user'], input['target_item']))
        }

        if cfg['update_best_model'] == 'local':
            self.pivot = [copy.deepcopy(self.pivot) for _ in range(cfg['num_nodes'])]

    def make_metric_name(self, metric_name):
        return metric_name

    def make_pivot(self):
        if cfg['data_name'] in ['ML1M', 'Anime']:
            if cfg['target_mode'] == 'explicit':
                pivot = float('inf')
                pivot_direction = 'down'
                pivot_name = 'RMSE'
            elif cfg['target_mode'] == 'implicit':
                pivot = -float('inf')
                pivot_direction = 'up'
                # pivot_name = 'MAP'
                pivot_name = 'NDCG'
            else:
                raise ValueError('Not valid target mode')
        else:
            raise ValueError('Not valid data name')
        return pivot, pivot_name, pivot_direction

    def evaluate(self, metric_names, input, output, node_idx=None):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        if node_idx:
            # if update_best_model == 'local',
            # add 'node_idx': node_idx to locate local model
            evaluation['node_idx'] = node_idx
        return evaluation

    def compare(self, val):
        if cfg['update_best_model'] == 'global':
            if self.pivot_direction == 'down':
                compared = self.pivot > val
            elif self.pivot_direction == 'up':
                compared = self.pivot < val
            else:
                raise ValueError('Not valid pivot direction')
            return compared
        elif cfg['update_best_model'] == 'local':
            '''
            compare and label the compare result of each node
            '''
            update_index_list = [False for _ in range(len(self.pivot))]
            for node_idx in val:
                compared = False
                if self.pivot_direction == 'down':
                    compared = self.pivot > val
                elif self.pivot_direction == 'up':
                    compared = self.pivot < val
                else:
                    raise ValueError('Not valid pivot direction')

                if compared:
                    update_index_list[node_idx] = True

            return update_index_list

    def update(self, val, update_index_list=None):
        if cfg['update_best_model'] == 'global':
            self.pivot = val
            return
        elif cfg['update_best_model'] == 'local':
            '''
            update the best result of each node
            '''
            for node_idx in range(len(update_index_list)):
                if update_index_list[node_idx] == True:
                    self.pivot[node_idx] = val[node_idx]
            return

        
