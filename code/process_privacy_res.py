import os
from config import cfg
import itertools
import json
import numpy as np
import pandas as pd
import argparse
import itertools

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
    # print('---controls', controls)
    return controls


def make_control_list(data, file):    
    if file == 'privacy_joint':
        controls = []
        # script_name = [['{}_privacy_joint.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            ml100k_controls = make_controls(control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            ml10m_controls = make_controls(control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            ml20m_controls = make_controls(control_name)
            controls.extend(ml20m_controls)
    elif file == 'privacy_federated_all':
        controls = []
        # script_name = [['{}_privacy_federated_all.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['100','300','max'], ['0'], ['l']]]
            ml100k_controls = make_controls(control_name)
            controls.extend(ml100k_controls)
        # if 'ML1M' in data:
        #     control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
        #                      ['0','1'], ['iid'], ['g'], ['100','300', 'max'], ['0'], ['l']]]
        #     ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
        #                                     device, control_name)
        #     controls.extend(ml1m_controls)
        # if 'Douban' in data:
        #     control_name = [[['Douban'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
        #                      ['0','1'], ['iid'], ['g'], ['100','300', 'max'], ['0'], ['l']]]
        #     douban_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
        #                                     device, control_name)
        #     controls.extend(douban_controls)

        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['100','300'], ['0'], ['l']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['100','300'], ['0'], ['l']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['0'], ['l']]]
            ml10m_controls = make_controls(control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['0'], ['l']]]
            ml20m_controls = make_controls(control_name)
            controls.extend(ml20m_controls)
        
    elif file == 'privacy_federated_decoder':
        controls = []
        # script_name = [['{}_privacy_federated_decoder.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['100','300','max'], ['1'], ['l']]]
            ml100k_controls = make_controls(control_name)
            controls.extend(ml100k_controls)
        # if 'ML1M' in data:
        #     control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
        #                      ['0','1'], ['iid'], ['g'], ['100','300', 'max'], ['1'], ['l']]]
        #     ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
        #                                     device, control_name)
        #     controls.extend(ml1m_controls)
        # if 'Douban' in data:
        #     control_name = [[['Douban'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
        #                      ['0','1'], ['iid'], ['g'], ['100','300', 'max'], ['1'], ['l']]]
        #     douban_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
        #                                     device, control_name)
        #     controls.extend(douban_controls)

        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['100','300'], ['1'], ['l']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'Douban' in data:
            control_name = [[['Douban'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
                             ['0','1'], ['iid'], ['g'], ['100','300'], ['1'], ['l']]]
            douban_controls = make_controls(control_name)
            controls.extend(douban_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['1'], ['l']]]
            ml10m_controls = make_controls(control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['1'], ['l']]]
            ml20m_controls = make_controls(control_name)
            controls.extend(ml20m_controls)
    else:
        raise ValueError('Not valid file')
    return

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

def process_result(controls):
    processed_result = {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result)
    summarize_result(processed_result)
    return processed_result

def process_result(controls):
    processed_result_exp, processed_result_history = {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    return processed_result_exp, processed_result_history

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

def make_learning_curve(processed_result):
    ylim_dict = {'iid': {'global': {'MNIST': [95, 100], 'CIFAR10': [50, 100], 'WikiText2': [0, 20]}},
                 'non-iid-2': {'global': {'MNIST': [50, 100], 'CIFAR10': [0, 70]},
                               'local': {'MNIST': [95, 100], 'CIFAR10': [50, 100]}}}
    fig = {}
    for exp_name in processed_result:
        control = exp_name.split('_')
        data_name = control[0]
        metric_name = metric_name_dict[data_name]
        control_name = control[-4]
        if control_name in ['a5-b5', 'a5-c5', 'a5-d5', 'a5-e5', 'a1-b1', 'a1-c1', 'a1-d1', 'a1-e1']:
            if 'non-iid-2' in exp_name:
                y = processed_result[exp_name]['Local-{}_mean'.format(metric_name)]
                x = np.arange(len(y))
                label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
                fig_name = '{}_lc_local'.format('_'.join(control[:-4] + control[-3:]))
                fig[fig_name] = plt.figure(fig_name)
                plt.plot(x, y, linestyle='-', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Communication rounds', fontsize=fontsize)
                plt.ylabel('Test {}'.format(metric_name), fontsize=fontsize)
                plt.ylim(ylim_dict['non-iid-2']['local'][data_name])
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
                y = processed_result[exp_name]['Global-{}_mean'.format(metric_name)]
                x = np.arange(len(y))
                label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
                fig_name = '{}_lc_global'.format('_'.join(control[:-4] + control[-3:]))
                fig[fig_name] = plt.figure(fig_name)
                plt.plot(x, y, linestyle='-', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Communication rounds', fontsize=fontsize)
                plt.ylabel('Test {}'.format(metric_name), fontsize=fontsize)
                plt.ylim(ylim_dict['non-iid-2']['global'][data_name])
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
            else:
                y = processed_result[exp_name]['Global-{}_mean'.format(metric_name)]
                x = np.arange(len(y))
                label_name = '-'.join(['{}'.format(x[0]) for x in list(control_name.split('-'))])
                fig_name = '{}_lc_global'.format('_'.join(control[:-4] + control[-3:]))
                fig[fig_name] = plt.figure(fig_name)
                plt.plot(x, y, linestyle='-', label=label_name)
                plt.legend(loc=loc_dict[data_name], fontsize=fontsize)
                plt.xlabel('Communication rounds', fontsize=fontsize)
                plt.ylabel('Test {}'.format(metric_name), fontsize=fontsize)
                plt.ylim(ylim_dict['iid']['global'][data_name])
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        plt.grid()
        fig_path = '{}/{}.{}'.format(vis_path, fig_name, cfg['save_format'])
        makedir_exist_ok(vis_path)
        plt.savefig(fig_path, dpi=500, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return

def main():
    # data = ['ML100K', 'ML1M', 'ML10M']
    data = ['ML1M', 'Douban']
    joint_control_list = make_control_list(data, 'privacy_joint')
    federated_all_control_list = make_control_list(data, 'privacy_federated_all')
    federated_decoder_control_list = make_control_list(data, 'privacy_federated_decoder')
    
    controls = joint_control_list + federated_all_control_list + federated_decoder_control_list
    processed_result_exp, processed_result_history = process_result(controls)
    with open('{}/processed_result_exp.json'.format(result_path), 'w') as fp:
        json.dump(processed_result_exp, fp, indent=2)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, processed_result_exp, [])
    df = make_df(extracted_processed_result)
    make_vis(df)
    extracted_processed_result = {}
    extract_processed_result(extracted_processed_result, processed_result_history, [])
    make_learning_curve(extracted_processed_result)
    return





if __name__ == '__main__':
    main()
