import copy

from collections import defaultdict
from collections.abc import Iterable
from email.policy import default
from torch.utils.tensorboard import SummaryWriter
from numbers import Number
from utils import ntuple, makedir_exist_ok
from config import cfg
import datetime


class Logger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.writer = None
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        self.history = defaultdict(list)
        self.iterator = defaultdict(int)
        self.mean_for_each_node = defaultdict(int)
        self.compress_activated_item_union_num_history = defaultdict(int)

    def reset(self):
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        self.mean_for_each_node = defaultdict(int)
        return

    def safe(self, write):
        if write:
            a = self.log_path
            self.writer = SummaryWriter(self.log_path)
            b = 5
        else:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            for name in self.mean:
                self.history[name].append(self.mean[name])
        return

    def append_compress_activated_item_union_num(self, total_activated_item_union_num, epoch):
        self.compress_activated_item_union_num_history[epoch] += total_activated_item_union_num
        return


    def append(self, result, tag, n=1, mean=True, is_fine_tune=False):
        """
        Append evaluation and average the evaluation dividing by total count

        Parameters:
            result - Torch. Evaluation of output['target_rating']
            tag - String. 'train' or 'test'

        Returns:
            None

        Raises:
            None
        """
        for k in result:
            name = '{}/{}'.format(tag, k)
            self.tracker[name] = result[k]
            if mean:
                if is_fine_tune:
                    if isinstance(result[k], Number):
                        self.counter[name] += n
                        self.mean[name] = ((self.counter[name] - n) * self.mean[name] + result[k]) / self.counter[name]
                else:
                    if cfg['update_best_model'] == 'global':
                        if isinstance(result[k], Number):
                            self.counter[name] += n
                            self.mean[name] = ((self.counter[name] - n) * self.mean[name] + n * result[k]) / self.counter[name]
                        elif isinstance(result[k], Iterable):
                            if name not in self.mean:
                                self.counter[name] = [0 for _ in range(len(result[k]))]
                                self.mean[name] = [0 for _ in range(len(result[k]))]
                            _ntuple = ntuple(len(result[k]))
                            n = _ntuple(n)
                            for i in range(len(result[k])):
                                self.counter[name][i] += n[i]
                                self.mean[name][i] = ((self.counter[name][i] - n[i]) * self.mean[name][i] + n[i] *
                                                    result[k][i]) / self.counter[name][i]
                        else:
                            raise ValueError('Not valid data type')
                    elif cfg['update_best_model'] == 'local':
                        node_idx = result['node_idx']
                        if name not in self.mean_for_each_node:
                            self.mean_for_each_node[name] = defaultdict(int)
                        if isinstance(result[k], Number):
                            self.mean_for_each_node[name][node_idx] = result[k]  
                    else:
                        raise ValueError('Not valid update_best_model way')
        return

    def write(self, tag, metric_names):
        """
        return the train/test log and evaluation result

        Parameters:
            tag - String. 'train' or 'test'
            metric_names - String. Pre-difinded metric

        Returns:
            info - String. The train/test log and evaluation result

        Raises:
            None
        """
        names = ['{}/{}'.format(tag, k) for k in metric_names]
        a = tag
        b = metric_names
        evaluation_info = []
        for name in names:
            tag, k = name.split('/')
            if isinstance(self.mean[name], Number):
                s = self.mean[name]
                evaluation_info.append('{}: {:.4f}'.format(k, s))
                if self.writer is not None:
                    self.iterator[name] += 1
                    self.writer.add_scalar(name, s, self.iterator[name])
            elif isinstance(self.mean[name], Iterable):
                s = tuple(self.mean[name])
                evaluation_info.append('{}: {}'.format(k, s))
                if self.writer is not None:
                    self.iterator[name] += 1
                    self.writer.add_scalar(name, s[0], self.iterator[name])
            else:
                raise ValueError('Not valid data type')
        info_name = '{}/info'.format(tag)
        info = self.tracker[info_name]
        info[2:2] = evaluation_info
        info = '  '.join(info)
        if self.writer is not None:
            self.iterator[info_name] += 1
            self.writer.add_text(info_name, info, self.iterator[info_name])
        return info

    def flush(self):
        self.writer.flush()
        return


def make_logger(path):
    logger_path = path
    makedir_exist_ok(logger_path)
    logger = Logger(logger_path)
    return logger
