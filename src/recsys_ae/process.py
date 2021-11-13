import os
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
save_format = 'pdf'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(data, file):
    if file == 'joint':
        controls = []
        control_name = [[data, ['user', 'item'], ['explicit', 'implicit'], ['base'], ['0']]]
        base_controls = make_controls(control_name)
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml100k_controls = make_controls(control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml10m_controls = make_controls(control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            ml20m_controls = make_controls(control_name)
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['ML20M'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'alone':
        controls = []
        control_name = [[data, ['user'], ['explicit', 'implicit'], ['base'], ['0'], ['genre', 'random-8']]]
        base_user_controls = make_controls(control_name)
        control_name = [[data, ['item'], ['explicit', 'implicit'], ['base'], ['0'], ['random-8']]]
        base_item_controls = make_controls(control_name)
        base_controls = base_user_controls + base_item_controls
        controls.extend(base_controls)
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['random-8']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['random-8']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['random-8']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['genre', 'random-8']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['random-8']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['mf', 'mlp', 'nmf', 'ae'],
                             ['0', '1'], ['random-8']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'assist':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'ar':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.3'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.3'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.3'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.3'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.3'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'aw':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['optim']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['optim']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['optim']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['optim']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['optim']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'ar-optim':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['optim-0.1'], ['constant']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['optim-0.1'], ['constant']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['optim-0.1'], ['constant']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['optim-0.1'], ['constant']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['optim-0.1'], ['constant']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    elif file == 'match':
        controls = []
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml100k_user_controls = make_controls(control_name)
            control_name = [[['ML100K'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml100k_item_controls = make_controls(control_name)
            ml100k_controls = ml100k_user_controls + ml100k_item_controls
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml1m_user_controls = make_controls(control_name)
            control_name = [[['ML1M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml1m_item_controls = make_controls(control_name)
            ml1m_controls = ml1m_user_controls + ml1m_item_controls
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml10m_user_controls = make_controls(control_name)
            control_name = [[['ML10M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml10m_item_controls = make_controls(control_name)
            ml10m_controls = ml10m_user_controls + ml10m_item_controls
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['genre', 'random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml20m_user_controls = make_controls(control_name)
            control_name = [[['ML20M'], ['item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            ml20m_item_controls = make_controls(control_name)
            ml20m_controls = ml20m_user_controls + ml20m_item_controls
            controls.extend(ml20m_controls)
        if 'NFP' in data:
            control_name = [[['NFP'], ['user', 'item'], ['explicit', 'implicit'], ['ae'], ['0', '1'],
                             ['random-8'], ['constant-0.1'], ['constant'], ['0.5']]]
            nfp_controls = make_controls(control_name)
            controls.extend(nfp_controls)
    else:
        raise ValueError('Not valid file')
    return controls


def main():
    data = ['ML100K', 'ML1M', 'ML10M']
    joint_control_list = make_control_list(data, 'joint')
    alone_control_list = make_control_list(data, 'alone')
    assist_control_list = make_control_list(data, 'assist')
    ar_control_list = make_control_list(data, 'ar')
    ar_optim_control_list = make_control_list(data, 'ar-optim')
    aw_epoch_control_list = make_control_list(data, 'aw')
    match_epoch_control_list = make_control_list(data, 'match')
    controls = joint_control_list + alone_control_list + assist_control_list + ar_control_list + \
               ar_optim_control_list + aw_epoch_control_list + match_epoch_control_list
    processed_result = process_result(controls)
    save(processed_result, os.path.join(result_path, 'processed_result.pt'))
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, processed_result, [])
    df = make_df(extracted_processed_result)
    make_vis(df)
    return


def process_result(controls):
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result)
    summarize_result(processed_result)
    return processed_result


def extract_result(control, model_tag, processed_result):
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            model_tag_list = model_tag.split('_')
            if len(model_tag_list) == 6:
                for k in base_result['logger']['test'].mean:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result:
                        processed_result[metric_name] = {'train': [None for _ in range(num_experiments)],
                                                         'test': [None for _ in range(num_experiments)]}
                    processed_result[metric_name]['train'][exp_idx] = base_result['logger']['train'].history[k]
                    processed_result[metric_name]['test'][exp_idx] = base_result['logger']['test'].mean[k]
            elif len(model_tag_list) == 7:
                for k in base_result['logger']['test'].mean:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result:
                        processed_result[metric_name] = {'train': [None for _ in range(num_experiments)],
                                                         'test': [None for _ in range(num_experiments)],
                                                         'test_each': [None for _ in range(num_experiments)]}
                    processed_result[metric_name]['train'][exp_idx] = base_result['logger']['train'].history[k]
                    processed_result[metric_name]['test'][exp_idx] = base_result['logger']['test'].mean[k]
                    processed_result[metric_name]['test_each'][exp_idx] = base_result['logger']['test_each'].history[k]
            elif len(model_tag_list) == 9:
                for k in base_result['logger']['test'].history:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result:
                        processed_result[metric_name] = {'train': [None for _ in range(num_experiments)],
                                                         'test': [None for _ in range(num_experiments)],
                                                         'test_each': [None for _ in range(num_experiments)],
                                                         'test_history': [None for _ in range(num_experiments)]}
                    processed_result[metric_name]['train'][exp_idx] = base_result['logger']['train'].history[k]
                    if metric_name in ['Loss', 'RMSE']:
                        processed_result[metric_name]['test'][exp_idx] = min(base_result['logger']['test'].history[k])
                        processed_result[metric_name]['test_each'][exp_idx] = \
                            np.array(base_result['logger']['test_each'].history[k]).reshape(-1, 11).min(axis=-1)
                    else:
                        processed_result[metric_name]['test'][exp_idx] = max(base_result['logger']['test'].history[k])
                        processed_result[metric_name]['test_each'][exp_idx] = \
                            np.array(base_result['logger']['test_each'].history[k]).reshape(-1, 11).max(axis=-1)
                    processed_result[metric_name]['test_history'][exp_idx] = base_result['logger']['test'].history[k]
            elif len(model_tag_list) == 10:
                for k in base_result['logger']['test'].history:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result:
                        processed_result[metric_name] = {'train': [None for _ in range(num_experiments)],
                                                         'test': [None for _ in range(num_experiments)],
                                                         'test_each': [None for _ in range(num_experiments)],
                                                         'test_history': [None for _ in range(num_experiments)]}
                    processed_result[metric_name]['train'][exp_idx] = base_result['logger']['train'].history[k]
                    if metric_name in ['Loss', 'RMSE']:
                        processed_result[metric_name]['test'][exp_idx] = min(base_result['logger']['test'].history[k])
                        processed_result[metric_name]['test_each'][exp_idx] = \
                            np.array(base_result['logger']['test_each'].history[k]).reshape(-1, 11).min(axis=-1)
                    else:
                        processed_result[metric_name]['test'][exp_idx] = max(base_result['logger']['test'].history[k])
                        processed_result[metric_name]['test_each'][exp_idx] = \
                            np.array(base_result['logger']['test_each'].history[k]).reshape(-1, 11).max(axis=-1)
                    processed_result[metric_name]['test_history'][exp_idx] = base_result['logger']['test'].history[k]
            else:
                raise ValueError('Not valid model tag')
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        if control[1] not in processed_result:
            processed_result[control[1]] = {}
            processed_result[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result[control[1]])
    return


def summarize_result(processed_result):
    pivot = ['train', 'test', 'test_each', 'test_history']
    leaf = False
    for k, v in processed_result.items():
        if k in pivot:
            leaf = True
            for x in processed_result[k]:
                if x is not None:
                    tmp = x
            for i in range(len(processed_result[k])):
                if processed_result[k][i] is None:
                    processed_result[k][i] = tmp
            e1 = [len(x) for x in processed_result[k] if isinstance(x, list)]
            flag = False
            for i in range(len(e1)):
                if e1[i] in [201, 12]:
                    if isinstance(processed_result[k][i], list):
                        # print(processed_result[k][i])
                        tmp_processed_result = None
                        for j in range(1, len(processed_result[k][i])):
                            if processed_result[k][i][j - 1] == processed_result[k][i][j]:
                                tmp_processed_result = processed_result[k][i][:j] + processed_result[k][i][j + 1:]
                        flag = True
                        if tmp_processed_result is not None:
                            processed_result[k][i] = tmp_processed_result
                if e1[i] > 12 and e1[i] < 200:
                    if isinstance(processed_result[k][i], list):
                        flag = True
                        tmp_processed_result = processed_result[k][i] + [processed_result[k][i][-1]] * (
                                    200 - len(processed_result[k][i]))
                        processed_result[k][i] = tmp_processed_result
                if e1[i] > 200:
                    if isinstance(processed_result[k][i], list):
                        flag = True
                        tmp_processed_result = processed_result[k][i][:200]
                        processed_result[k][i] = tmp_processed_result
            e2 = [len(x) for x in processed_result[k] if isinstance(x, list)]
            # print(k, e1, e2)
            stacked_result = np.stack(processed_result[k], axis=0)
            processed_result[k] = {}
            processed_result[k]['mean'] = np.mean(stacked_result, axis=0)
            processed_result[k]['std'] = np.std(stacked_result, axis=0)
            processed_result[k]['max'] = np.max(stacked_result, axis=0)
            processed_result[k]['min'] = np.min(stacked_result, axis=0)
            processed_result[k]['argmax'] = np.argmax(stacked_result, axis=0)
            processed_result[k]['argmin'] = np.argmin(stacked_result, axis=0)
            processed_result[k]['val'] = stacked_result.tolist()
    if not leaf:
        for k, v in processed_result.items():
            # print(k)
            summarize_result(v)
        return
    return


def extract_processed_result(extracted_processed_result, processed_result, control):
    pivot = ['train', 'test', 'test_each', 'test_history']
    leaf = False
    for k, v in processed_result.items():
        if k in pivot:
            leaf = True
            exp_name = '_'.join(control[:-1])
            metric_name = control[-1]
            if exp_name not in extracted_processed_result:
                extracted_processed_result[exp_name] = {p: defaultdict() for p in processed_result.keys()}
            extracted_processed_result[exp_name][k]['{}_mean'.format(metric_name)] = processed_result[k]['mean']
            extracted_processed_result[exp_name][k]['{}_std'.format(metric_name)] = processed_result[k]['std']
    if not leaf:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def make_df(extracted_processed_result):
    pivot = ['train', 'test', 'test_each', 'test_history']
    df = {p: defaultdict(list) for p in pivot}
    for exp_name in extracted_processed_result:
        control = exp_name.split('_')
        for p in extracted_processed_result[exp_name]:
            if len(control) == 5:
                data_name, data_mode, target_mode, model_name, info = control
                index_name = [model_name]
                df_name = '_'.join([data_name, data_mode, target_mode, info])
            elif len(control) == 6:
                data_name, data_mode, target_mode, model_name, info, data_split_mode = control
                index_name = [model_name]
                df_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode])
            elif len(control) == 8:
                data_name, data_mode, target_mode, model_name, info, data_split_mode, ar, aw = control
                index_name = ['_'.join([model_name, ar, aw])]
                df_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode, 'assist'])
            elif len(control) == 9:
                data_name, data_mode, target_mode, model_name, info, data_split_mode, ar, aw, match_rate = control
                index_name = ['_'.join([model_name, ar, aw, match_rate])]
                df_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode, 'assist'])
            else:
                raise ValueError('Not valid control')
            metric = list(extracted_processed_result[exp_name][p].keys())
            if len(metric) > 0:
                if len(extracted_processed_result[exp_name][p][metric[0]].shape) == 0:
                    df[p][df_name].append(pd.DataFrame(data=extracted_processed_result[exp_name][p], index=index_name))
                else:
                    for m in metric:
                        df_name_ = '{}_{}'.format(df_name, m)
                        df[p][df_name_].append(
                            pd.DataFrame(data=extracted_processed_result[exp_name][p][m].reshape(1, -1),
                                         index=index_name))
    for p in pivot:
        startrow = 0
        writer = pd.ExcelWriter('{}/{}.xlsx'.format(result_path, p), engine='xlsxwriter')
        for df_name in df[p]:
            df[p][df_name] = pd.concat(df[p][df_name])
            df[p][df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[p][df_name].index) + 3
        writer.save()
    return df


def make_vis(df):
    control_dict = {'Joint': 'Joint', 'constant-0.1_constant': 'AAE ($\eta_k=0.1$)',
                    'constant-0.3_constant': 'AAE ($\eta_k=0.3$)', 'optim-0.1_constant': 'AAE (Optimize $\eta_k$)',
                    'constant-0.1_optim': 'AAE (Optimize $w_k$)', 'constant-0.1_constant_0.5': 'AAE ($50\%$ alignment)',
                    'Alone': 'Alone'}
    color = {'Joint': 'black', 'constant-0.1_constant': 'red', 'constant-0.3_constant': 'orange',
             'optim-0.1_constant': 'dodgerblue', 'constant-0.1_optim': 'blue', 'constant-0.1_constant_0.5': 'purple',
             'Alone': 'green'}
    linestyle = {'Joint': '-', 'constant-0.1_constant': '--', 'constant-0.3_constant': ':', 'optim-0.1_constant': '-.',
                 'constant-0.1_optim': (0, (1, 5)), 'constant-0.1_constant_0.5': (0, (5, 1)), 'Alone': (0, (5, 5))}
    loc_dict = {'Loss': 'upper right', 'RMSE': 'upper right', 'Accuracy': 'lower right', 'MAP': 'lower right'}
    fontsize = {'legend': 12, 'label': 16, 'ticks': 16}
    linewidth = 3
    fig = {}
    p = 'test_history'
    for df_name in df[p]:
        df_name_list = df_name.split('_')
        if 'assist' in df_name_list:
            data_name, data_mode, target_mode, info, data_split_mode, assist, metric_name, stat = df_name.split('_')
            if stat == 'std':
                continue
            df_name_std = '_'.join(
                [data_name, data_mode, target_mode, info, data_split_mode, assist, metric_name, 'std'])
            df_name_joint = '_'.join([data_name, data_mode, target_mode, info])
            df_name_alone = '_'.join([data_name, data_mode, target_mode, info, data_split_mode])
            fig_name = '_'.join([data_name, data_mode, target_mode, info, data_split_mode, assist, metric_name])
            fig[fig_name] = plt.figure(fig_name)
            for ((index, row), (_, row_std)) in zip(df[p][df_name].iterrows(), df[p][df_name_std].iterrows()):
                model_name = index.split('_')[0]
                control = '_'.join(index.split('_')[1:])
                y = row.to_numpy()
                y_err = row_std.to_numpy()
                x = np.arange(len(y))
                plt.plot(x, y, color=color[control], linestyle=linestyle[control], label=control_dict[control],
                         linewidth=linewidth)
                plt.fill_between(x, (y - y_err), (y + y_err), color=color[control], alpha=.1)
                plt.xlabel('Assistance Rounds', fontsize=fontsize['label'])
                plt.ylabel(metric_name, fontsize=fontsize['label'])
                plt.xticks(fontsize=fontsize['ticks'])
                plt.yticks(fontsize=fontsize['ticks'])
            y_joint = df['test'][df_name_joint]['{}_{}'.format(metric_name, stat)].loc[model_name]
            y_joint = np.full(x.shape, y_joint)
            y_err_joint = df['test'][df_name_joint]['{}_std'.format(metric_name)].loc[model_name]
            plt.plot(x, y_joint, color=color['Joint'], linestyle=linestyle['Joint'], label=control_dict['Joint'],
                     linewidth=linewidth)
            plt.fill_between(x, (y_joint - y_err_joint), (y_joint + y_err_joint), color=color['Joint'], alpha=.1)
            y_alone = df['test'][df_name_alone]['{}_{}'.format(metric_name, stat)].loc[model_name]
            y_alone = np.full(x.shape, y_alone)
            y_err_alone = df['test'][df_name_joint]['{}_std'.format(metric_name)].loc[model_name]
            plt.plot(x, y_alone, color=color['Alone'], linestyle=linestyle['Alone'], label=control_dict['Alone'],
                     linewidth=linewidth)
            plt.fill_between(x, (y_alone - y_err_alone), (y_alone + y_err_alone), color=color['Alone'], alpha=.1)
            plt.legend(loc=loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, save_format)
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


if __name__ == '__main__':
    main()
