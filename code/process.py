import os

from train_privacy_federated_all import train

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import itertools
import json
import numpy as np
import pandas as pd
from utils import save, load, makedir_exist_ok
import matplotlib.pyplot as plt
from collections import defaultdict

result_path = './output/result'
save_format = 'png'
vis_path = './output/vis/{}'.format(save_format)
num_experiments = 4
exp = [str(x) for x in list(range(num_experiments))]
# exp = ['1']
dpi = 300


def make_controls(control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    controls = [exp] + [control_names]
    controls = list(itertools.product(*controls))
    return controls


def make_control_list(data, file):
    if file == 'privacy_joint':
        controls = []
        # control_name = [[data, ['user', 'item'], ['explicit', 'implicit'], ['base'], ['0']]]
        # base_controls = make_controls(control_name)
        # controls.extend(base_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'Anime' in data:
            control_name = [[['Anime'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            anime_controls = make_controls(control_name)
            controls.extend(anime_controls)
        if 'Netflix' in data:
            control_name = [[['Netflix'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0'], ['iid'], ['g'], ['1'], ['0'], ['l']]]
            netflix_controls = make_controls(control_name)
            controls.extend(netflix_controls)
    elif file == 'privacy_federated_all':
        controls = []
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', '300', 'max'], ['0'], ['l']]]
            # control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
            #                  ['0'], ['iid'], ['g'], ['100'], ['0'], ['l']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'Anime' in data:
            control_name = [[['Anime'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['0'], ['l']]]
            anime_controls = make_controls(control_name)
            controls.extend(anime_controls)
        if 'Netflix' in data:
            control_name = [[['Netflix'], ['user'], ['ex','im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['0'], ['l']]]
            netflix_controls = make_controls(control_name)
            controls.extend(netflix_controls)
    elif file == 'privacy_federated_decoder':
        controls = []
        if 'ML1M' in data:
            # control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
            #                  ['0'], ['iid'], ['g'], ['100', '300', 'max'], ['1'], ['l']]]
            control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', '300', 'max'], ['1'], ['l']]]
            ml1m_controls = make_controls(control_name)
            controls.extend(ml1m_controls)
        if 'Anime' in data:
            control_name = [[['Anime'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['1'], ['l']]]
            anime_controls = make_controls(control_name)
            controls.extend(anime_controls)
        if 'Netflix' in data:
            control_name = [[['Netflix'], ['user'], ['ex','im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100','300'], ['1'], ['l']]]
            netflix_controls = make_controls(control_name)
            controls.extend(netflix_controls)
    else:
        raise ValueError('Not valid file')
    return controls


def extract_result(control, model_tag, processed_result_exp, processed_result_history, processed_result_each, processed_result_each_compress_ratio):
    a = control
    if len(control) == 1:
        exp_idx = exp.index(control[0])
        base_result_path_i = os.path.join(result_path, '{}.pt'.format(model_tag))
        if os.path.exists(base_result_path_i):
            base_result = load(base_result_path_i)
            if 'de' in model_tag:

                best_train_epoch = base_result['epoch'] - 2
                train_logger = base_result['logger']['train']
                compress_parameter_ratio_per_epoch = base_result['compress_parameter_ratio_per_epoch']
                
                train_logger_history = train_logger.history 
                # for i in range(len(train_logger_history['test/NDCG'])):
                #     print(i, train_logger_history['test/NDCG'][i])               
                for k in train_logger_history:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                        processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_each[metric_name] = {'each': [None for _ in range(num_experiments)]}
                        
                    processed_result_exp[metric_name]['exp'][exp_idx] = train_logger.history[k][best_train_epoch]
                    processed_result_history[metric_name]['history'][exp_idx] = train_logger.history[k]
                    processed_result_each[metric_name]['each'][exp_idx] = train_logger.history[k]
                
                compress_parameter_ratio_per_epoch_list = []
                for key, val in compress_parameter_ratio_per_epoch.items():
                    compress_parameter_ratio_per_epoch_list.append(1/val*2)
                if 'compress_ratio' not in processed_result_each_compress_ratio:
                    processed_result_each_compress_ratio['compress_ratio'] = [None for _ in range(num_experiments)]
                processed_result_each_compress_ratio['compress_ratio'][exp_idx] = np.array(compress_parameter_ratio_per_epoch_list)
                
                # for k in base_result['logger']['test'].history:
                #     metric_name = k.split('/')[1]
                #     if metric_name not in processed_result_exp:
                #         processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                #         processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                #         processed_result_each[metric_name] = {'each': [None for _ in range(num_experiments)]}
                #     if metric_name in ['NDCG']:
                #         test = max(base_result['logger']['test'].history[k])
                #         history = base_result['logger']['test'].history[k]
                #         test_each = base_result['logger']['test_each'].history[k]
                #         if len(test_each) % 11 > 0:
                #             print(control, model_tag)
                #             continue
                #         test_each = np.amax(np.array(test_each).reshape(11, -1), axis=0).tolist()
                #     else:
                #         test = min(base_result['logger']['test'].history[k])
                #         history = base_result['logger']['test'].history[k]
                #         test_each = base_result['logger']['test_each'].history[k]
                #         if len(test_each) % 11 > 0:
                #             print(control, model_tag)
                #             continue
                #         test_each = np.amin(np.array(test_each).reshape(11, -1), axis=0).tolist()
                #     processed_result_exp[metric_name]['exp'][exp_idx] = test
                #     processed_result_history[metric_name]['history'][exp_idx] = history
                #     processed_result_each[metric_name]['each'][exp_idx] = test_each
            else:
                for k in base_result['logger']['test'].mean:
                    metric_name = k.split('/')[1]
                    if metric_name not in processed_result_exp:
                        processed_result_exp[metric_name] = {'exp': [None for _ in range(num_experiments)]}
                        processed_result_history[metric_name] = {'history': [None for _ in range(num_experiments)]}
                        processed_result_each[metric_name] = {'each': [None for _ in range(num_experiments)]}
                    
                    best_train_epoch = base_result['epoch'] - 2
                    # print(f'k: {k}')
                    
                    history_local = base_result['logger']['train'].history[k]
                    # print('1111', history_local, len(history_local))
                    if type(history_local) == list and len(history_local) < 800:
                        supplement = [history_local[-1]] * (800 - len(history_local))
                        history_local.extend(supplement)
                    # print('22222', history_local)
                    # if 'all' in model_tag:
                    exp_local = history_local[best_train_epoch]
                    
                    each_local = history_local
                    # else:
                    #     exp_local = base_result['logger']['test'].mean[k]
                    #     history_local = base_result['logger']['train'].history[k]
                    #     each_local = base_result['logger']['test'].history[k]

                    # for i in range(len(base_result['logger']['train'].history[k])):
                    #     print([(i, base_result['logger']['train'].history[k][i]) for i in range(len(base_result['logger']['train'].history[k]))])  

                    # if type(exp_local) == list and len(exp_local) < 800:
                    #     supplement = [exp_local[-1]] * (800 - len(exp_local))
                    #     exp_local = exp_local.extend(supplement)
                    
                    # if type(each_local) == list and len(each_local) < 800:
                    #     supplement = [each_local[-1]] * (800 - len(each_local))
                    #     each_local = each_local.extend(supplement)

                    processed_result_exp[metric_name]['exp'][exp_idx] = exp_local
                    processed_result_history[metric_name]['history'][exp_idx] = history_local
                    processed_result_each[metric_name]['each'][exp_idx] = each_local
        else:
            print('Missing {}'.format(base_result_path_i))
    else:
        # print('extract_result len(control) is not 1')
        if control[1] not in processed_result_exp:
            processed_result_exp[control[1]] = {}
            processed_result_history[control[1]] = {}
            processed_result_each[control[1]] = {}
            processed_result_each_compress_ratio[control[1]] = {}
        extract_result([control[0]] + control[2:], model_tag, processed_result_exp[control[1]],
                       processed_result_history[control[1]], processed_result_each[control[1]], processed_result_each_compress_ratio[control[1]])
        # extract_result(control, model_tag, processed_result_exp, processed_result_history, processed_result_each, processed_result_each_compress_ratio):
    return


def summarize_result(processed_result):
    for pivot in list(processed_result.keys()):
        if pivot == 'exp':
            processed_result[pivot] = [x for x in processed_result[pivot] if x is not None]
            processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
            processed_result['mean'] = np.mean(processed_result[pivot], axis=0).item()
            processed_result['std'] = np.std(processed_result[pivot], axis=0).item()
            processed_result['max'] = np.max(processed_result[pivot], axis=0).item()
            processed_result['min'] = np.min(processed_result[pivot], axis=0).item()
            processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0).item()
            processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0).item()
            processed_result[pivot] = processed_result[pivot].tolist()
        elif pivot in ['history', 'each', 'compress_ratio']:
            processed_result[pivot] = [x for x in processed_result[pivot] if x is not None]
            for i in range(len(processed_result[pivot])):
                if len(processed_result[pivot][i]) == 201:
                    processed_result[pivot][i] = processed_result[pivot][i][:200]
            processed_result[pivot] = np.stack(processed_result[pivot], axis=0)
            processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
            if pivot in ['compress_ratio']:
                processed_result['mean'] = np.mean(processed_result[pivot], axis=0)
            processed_result['std'] = np.std(processed_result[pivot], axis=0)
            processed_result['max'] = np.max(processed_result[pivot], axis=0)
            processed_result['min'] = np.min(processed_result[pivot], axis=0)
            processed_result['argmax'] = np.argmax(processed_result[pivot], axis=0)
            processed_result['argmin'] = np.argmin(processed_result[pivot], axis=0)
            processed_result[pivot] = processed_result[pivot].tolist()
        else:
            for k, v in processed_result.items():
                summarize_result(processed_result=v)
            return
    return

def process_result(controls):
    processed_result_exp, processed_result_history, processed_result_each, processed_result_each_compress_ratio = {}, {}, {}, {}
    for control in controls:
        model_tag = '_'.join(control)
        extract_result(list(control), model_tag, processed_result_exp, processed_result_history, processed_result_each, processed_result_each_compress_ratio)
    summarize_result(processed_result_exp)
    summarize_result(processed_result_history)
    summarize_result(processed_result_each)
    if processed_result_each_compress_ratio != {}:
        summarize_result(processed_result_each_compress_ratio)
    return processed_result_exp, processed_result_history, processed_result_each, processed_result_each_compress_ratio

def extract_processed_result(extracted_processed_result, processed_result, control):
    if 'exp' in processed_result or 'history' in processed_result or 'each' in processed_result or 'compress_ratio' in processed_result:
        mode_name = '_'.join(control[:-1])
        metric_name = control[-1]
        if mode_name not in extracted_processed_result:
            extracted_processed_result[mode_name] = defaultdict()
        extracted_processed_result[mode_name]['{}_mean'.format(metric_name)] = processed_result['mean']
        extracted_processed_result[mode_name]['{}_std'.format(metric_name)] = processed_result['std']
    else:
        for k, v in processed_result.items():
            extract_processed_result(extracted_processed_result, v, control + [k])
    return


def main():
    write = True
    # data = ['ML1M', 'Douban', 'Amazon']
    # mode = ['joint', 'alone', 'mdr', 'assist', 'match', 'match-mdr', 'info', 'pl', 'cs', 'cs-alone', 'cs-mdr', 'aw']
    # data = ['ML1M', 'Anime', 'Netflix']
    # file = ['privacy_joint', 'privacy_federated_all', 'privacy_federated_decoder']
    data = ['Anime', 'ML1M']
    # data = ['Anime']
    file = ['privacy_joint', 'privacy_federated_all', 'privacy_federated_decoder']

    # file = ['privacy_federated_all']

    controls = []
    for data_ in data:
        for file_ in file:
            controls += make_control_list(data_, file_)
    
    # print(f'---controls--: {controls}')
    processed_result_exp, processed_result_history, processed_result_each, processed_result_each_compress_ratio = process_result(controls)
    save(processed_result_exp, os.path.join(result_path, 'processed_result_exp.pt'))
    save(processed_result_history, os.path.join(result_path, 'processed_result_history.pt'))
    save(processed_result_each, os.path.join(result_path, 'processed_result_each.pt'))
    # only for decoder
    if processed_result_each_compress_ratio != {}:
        save(processed_result_each, os.path.join(result_path, 'processed_result_each_compress_ratio.pt'))
    extracted_processed_result_exp = {}
    extracted_processed_result_history = {}
    extracted_processed_result_each = {}
    extracted_processed_result_each_compress_ratio = {}
    extract_processed_result(extracted_processed_result_exp, processed_result_exp, [])
    extract_processed_result(extracted_processed_result_history, processed_result_history, [])
    extract_processed_result(extracted_processed_result_each, processed_result_each, [])
    if processed_result_each_compress_ratio != {}:
        extract_processed_result(extracted_processed_result_each_compress_ratio, processed_result_each_compress_ratio, [])
    
    df_exp = make_df_result(extracted_processed_result_exp, 'exp', write)
    df_history = make_df_result(extracted_processed_result_history, 'history', write)
    df_each = make_df_result(extracted_processed_result_each, 'each', write)
    if processed_result_each_compress_ratio != {}:
        df_each_compress_ratio = make_df_result(extracted_processed_result_each_compress_ratio, 'each_compress_ratio', write)
    make_vis_lc(df_exp, df_history)
    if processed_result_each_compress_ratio != {}:
        make_vis_compress_ratio(df_each_compress_ratio)
    # make_vis_lc_best(df_exp, df_history)
    # make_vis_match(df_exp)
    # make_vis_cs(df_exp, df_each)
    return

def make_vis_compress_ratio(df_each_compress_ratio):
    joint_key = 'Joint'
    nodes_100 = '100 Clients'
    nodes_300 = '300 Clients'
    nodes_max = '1 User / Client'
    control_dict = {joint_key: joint_key, nodes_100: nodes_100, nodes_300: nodes_300, nodes_max: nodes_max}
    color_dict = {joint_key: 'black', nodes_100: 'gray', nodes_300: 'green', nodes_max: 'pink'}
    linestyle_dict = {joint_key: '-', nodes_100: ':', nodes_300: (5, (1, 5)), nodes_max: (5, (5, 5))}
    marker_dict = {joint_key: 'X', nodes_100: 'x', nodes_300: 'p', nodes_max: '^'}

    # joint_pic_key = 'Joint'
    # federated_all_pic_key = 'FL'
    # federated_de_pic_key = 'PFL'
    # # nodes_max = '1 User/Node'
    # control_dict = {joint_pic_key: joint_pic_key, federated_all_pic_key: federated_all_pic_key, federated_de_pic_key: federated_de_pic_key}
    # color_dict = {joint_pic_key: 'black', federated_all_pic_key: 'green', federated_de_pic_key: 'pink'}
    # linestyle_dict = {joint_pic_key: '-', federated_all_pic_key: '-', federated_de_pic_key: '-'}

    # linestyle_dict = {joint_pic_key: (5, (1, 5)), federated_all_pic_key: (5, (1, 5)), federated_de_pic_key: (5, (1, 5))}
    

    # marker_dict = {joint_pic_key: 'X', federated_all_pic_key: 'p', federated_de_pic_key: '^'}
    # label_loc_dict = {'Loss': 'upper right', 'RMSE': 'upper right', 'NDCG': 'lower right'}
    label_loc_dict = {'Compress Ratio': 'upper right'}
    fontsize = {'legend': 10, 'label': 10, 'ticks': 10}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}

    ML1M_dict = {}
    Anime_dict = {}
    Netflix_dict = {}

    ML1M_std_dict = {}
    Anime_std_dict = {}
    Netflix_std_dict = {}
    for df_name in df_each_compress_ratio:
        df_name_list = df_name.split('_')
        # info = df_name_list[3]
        # metric_name = df_name_list[-2]
        stat = df_name_list[-1]
        # valid_mask = info == '0' and metric_name in ['RMSE', 'NDCG'] and stat == 'mean'
        valid_mask = stat == 'mean'
        if valid_mask:
            # data_name, data_mode, target_mode, info, data_split_mode, metric_name, stat = df_name_list
            # data_name, data_mode, target_mode, metric_name, stat = df_name_list
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            
        

            # alone = {}
            # mdr = {}
            for (index, row) in df_each_compress_ratio[df_name].iterrows():
                # 区分all/de, 100/300(nodes数量)
                # index_list = index.split('_')
                # nodes_count = index_list[-4]
                if 'ML1M' in df_name:
                    tem_ = row.to_numpy()
                    ML1M_dict[df_name] = tem_[~np.isnan(tem_)]
                elif 'Anime' in df_name:
                    tem_ = row.to_numpy()
                    Anime_dict[df_name] = tem_[~np.isnan(tem_)]
                elif 'Netflix' in df_name:
                    tem_ = row.to_numpy()
                    Netflix_dict[df_name] = tem_[~np.isnan(tem_)]
                else:
                    raise ValueError('data wrong')

            for (index, row) in df_each_compress_ratio[df_name_std].iterrows():
                # index_list = index.split('_')
                if 'ML1M' in df_name_std:
                    tem_ = row.to_numpy()
                    ML1M_std_dict[df_name_std] = tem_[~np.isnan(tem_)]
                elif 'Anime' in df_name_std:
                    tem_ = row.to_numpy()
                    Anime_std_dict[df_name_std] = tem_[~np.isnan(tem_)]
                elif 'Netflix' in df_name_std:
                    tem_ = row.to_numpy()
                    Netflix_std_dict[df_name_std] = tem_[~np.isnan(tem_)]
                else:
                    raise ValueError('data wrong')


            # joint_values = np.array(list(joint.values())).reshape(-1)
            # joint_std_values = np.array(list(joint_std.values())).reshape(-1)
            # if metric_name in ['NDCG']:
            #     joint_best_idx = np.argmax(joint_values)
            #     # alone_best_idx = np.argmax(alone_values)
            #     # mdr_best_idx = np.argmax(mdr_values)
            # else:
            #     joint_best_idx = np.argmin(joint_values)
            #     # alone_best_idx = np.argmin(alone_values)
            #     # mdr_best_idx = np.argmin(mdr_values)
            
            # joint = joint_values[joint_best_idx]
            # joint_std = joint_std_values[joint_best_idx]
            # joint = np.full(x.shape, joint)
            # joint_std = np.full(x.shape, joint_std)

            x = np.arange(800)

    # 分figure
    for item in [(ML1M_dict, ML1M_std_dict), (Anime_dict, Anime_std_dict)]:
        sub_dict = item[0]
        sub_std_dict = item[1]
        for key in sub_dict.keys():
            key_list = key.split('_')
            if 'im' in key_list:
                break
            fig_name = '_'.join([key_list[1]])
            # fig_name = '_'.join([*df_name_list[:-1]])
            if fig_name not in fig:
                fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]

            # joint_key = 'Joint'
            # nodes_100 = '100 Nodes'
            # nodes_300 = '300 Nodes'
            # nodes_max = '1 User/Node'

            if '100' in key:
                control = nodes_100
            elif '300' in key:
                control = nodes_300
            elif 'max' in key:
                control = nodes_max
            else:
                raise ValueError('node number error')

            std_key = key.replace('mean', 'std')
            # ax_1.errorbar(x, sub_dict[key], yerr=sub_std_dict[std_key], color=color_dict[control], linestyle=linestyle_dict[control],
            #             label=control_dict[control])
            ax_1.errorbar(x, sub_dict[key], yerr=None, color=color_dict[control], linestyle=linestyle_dict[control],
                        label=control_dict[control])
            y = sub_dict[key]
            y_err = sub_std_dict[std_key]
            ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[control], alpha=.1)

        # ax_1.set_xticks(x)
        # ax_1.set_xticks()
        # if sub_dict != {}:
        print('aaaaa')
        ax_1.set_xlabel('Communication Rounds', fontsize=fontsize['label'])
        ax_1.set_ylabel('Compress Ratio', fontsize=fontsize['label'])
        ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
        ax_1.legend(loc=label_loc_dict['Compress Ratio'], fontsize=fontsize['legend'])
        

    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.1')
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'lc', *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return



def make_df_result(extracted_processed_result, mode_name, write):
    df = defaultdict(list)
    for exp_name in extracted_processed_result:
        control = exp_name.split('_')
        for metric_name in extracted_processed_result[exp_name]:
            df_name = '_'.join([*control[:3], metric_name])
            index_name = '_'.join([*control[:]])
            df[df_name].append(pd.DataFrame(data=[extracted_processed_result[exp_name][metric_name]],
                                            index=[index_name]))
    if write:
        startrow = 0
        writer = pd.ExcelWriter('{}/result_{}.xlsx'.format(result_path, mode_name), engine='xlsxwriter')
        for df_name in df:
            concat_df = pd.concat(df[df_name])
            df[df_name] = concat_df
            df[df_name].to_excel(writer, sheet_name='Sheet1', startrow=startrow + 1)
            writer.sheets['Sheet1'].write_string(startrow, 0, df_name)
            startrow = startrow + len(df[df_name].index) + 3
        writer.save()
    else:
        for df_name in df:
            df[df_name] = pd.concat(df[df_name])
    return df

def make_vis_lc(df_exp, df_history):
    # joint_key = 'Joint'
    # nodes_100 = '100 Nodes'
    # nodes_300 = '300 Nodes'
    # nodes_max = '1 User/Node'
    # control_dict = {joint_key: joint_key, nodes_100: nodes_100, nodes_300: nodes_300, nodes_max: nodes_max}
    # color_dict = {joint_key: 'black', nodes_100: 'gray', nodes_300: 'green', nodes_max: 'pink'}
    # linestyle_dict = {joint_key: '-', nodes_100: ':', nodes_300: (5, (1, 5)), nodes_max: (5, (5, 5))}
    # marker_dict = {joint_key: 'X', nodes_100: 'x', nodes_300: 'p', nodes_max: '^'}

    joint_pic_key = 'Joint'
    federated_all_pic_key = 'FedAvg'
    federated_de_pic_key = 'PersonalFR'
    # nodes_max = '1 User/Node'
    control_dict = {joint_pic_key: joint_pic_key, federated_all_pic_key: federated_all_pic_key, federated_de_pic_key: federated_de_pic_key}
    color_dict = {joint_pic_key: 'black', federated_all_pic_key: 'green', federated_de_pic_key: 'red'}
    linestyle_dict = {joint_pic_key: '-', federated_all_pic_key: ':', federated_de_pic_key: (5, (5, 5))}

    # linestyle_dict = {joint_pic_key: (5, (1, 5)), federated_all_pic_key: (5, (1, 5)), federated_de_pic_key: (5, (1, 5))}
    

    # marker_dict = {joint_pic_key: 'X', federated_all_pic_key: 'p', federated_de_pic_key: '^'}
    label_loc_dict = {'Loss': 'upper right', 'RMSE': 'upper right', 'NDCG': 'lower right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        # info = df_name_list[3]
        metric_name = df_name_list[-2]
        stat = df_name_list[-1]
        # valid_mask = info == '0' and metric_name in ['RMSE', 'NDCG'] and stat == 'mean'
        valid_mask = metric_name in ['RMSE', 'NDCG'] and stat == 'mean'
        if valid_mask:
            # data_name, data_mode, target_mode, info, data_split_mode, metric_name, stat = df_name_list
            data_name, data_mode, target_mode, metric_name, stat = df_name_list
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            joint = {}
            federated_all = {}
            federated_de = {}

            # alone = {}
            # mdr = {}
            for (index, row) in df_exp[df_name].iterrows():
                # 区分all/de, 100/300(nodes数量)
                index_list = index.split('_')
                if 'joint' in index_list:
                    joint_ = row.to_numpy()
                    joint[index] = joint_[~np.isnan(joint_)]

            for (index, row) in df_history[df_name].iterrows():
                # 区分all/de, 100/300(nodes数量)
                index_list = index.split('_')
                # if 'joint' in index_list:
                #     joint_ = row.to_numpy()
                #     joint[index] = joint_[~np.isnan(joint_)]
                if 'all' in index_list:
                    all_ = row.to_numpy()
                    federated_all[index] = all_[~np.isnan(all_)]
                elif 'de' in index_list:
                    de_ = row.to_numpy()
                    federated_de[index] = de_[~np.isnan(de_)]
                # else:
                #     raise ValueError('data wrong')


            joint_std = {}
            federated_all_std = {}
            federated_de_std = {}
            for (index, row) in df_exp[df_name_std].iterrows():
                index_list = index.split('_')
                if 'joint' in index_list:
                    joint_ = row.to_numpy()
                    joint_std[index] = joint_[~np.isnan(joint_)]

            for (index, row) in df_history[df_name_std].iterrows():
                index_list = index.split('_')
                # if 'joint' in index_list:
                #     joint_ = row.to_numpy()
                #     joint_std[index] = joint_[~np.isnan(joint_)]
                if 'all' in index_list:
                    all_ = row.to_numpy()
                    federated_all_std[index] = all_[~np.isnan(all_)]
                elif 'de' in index_list:
                    de_ = row.to_numpy()
                    federated_de_std[index] = de_[~np.isnan(de_)]
                # else:
                #     raise ValueError('data wrong')


            joint_values = np.array(list(joint.values())).reshape(-1)
            joint_std_values = np.array(list(joint_std.values())).reshape(-1)
            if metric_name in ['NDCG']:
                joint_best_idx = np.argmax(joint_values)
                # alone_best_idx = np.argmax(alone_values)
                # mdr_best_idx = np.argmax(mdr_values)
            else:
                joint_best_idx = np.argmin(joint_values)
                # alone_best_idx = np.argmin(alone_values)
                # mdr_best_idx = np.argmin(mdr_values)
            x = np.arange(800)
            joint = joint_values[joint_best_idx]
            joint_std = joint_std_values[joint_best_idx]
            joint = np.full(x.shape, joint)
            joint_std = np.full(x.shape, joint_std)

            # 分figure
            for key in federated_de.keys():
                key_list = key.split('_')
                fig_name = '_'.join([key_list[0], key_list[2], key_list[-3]])
                # fig_name = '_'.join([*df_name_list[:-1]])
                fig[fig_name] = plt.figure(fig_name, figsize=figsize)
                if fig_name not in ax_dict_1:
                    ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
                ax_1 = ax_dict_1[fig_name]

                control = joint_pic_key
                # ax_1.errorbar(x, joint, yerr=joint_std, color=color_dict[control], linestyle=linestyle_dict[control],
                #             label=control_dict[control])
                ax_1.errorbar(x, joint, yerr=None, color=color_dict[control], linestyle=linestyle_dict[control],
                            label=control_dict[control])
                ax_1.fill_between(x, (joint - joint_std), (joint + joint_std), color=color_dict[control], alpha=.1)


                control = federated_de_pic_key
                # ax_1.errorbar(x, federated_de[key], yerr=federated_de_std[key], color=color_dict[control], linestyle=linestyle_dict[control],
                #             label=control_dict[control])
                ax_1.errorbar(x, federated_de[key], yerr=None, color=color_dict[control], linestyle=linestyle_dict[control],
                            label=control_dict[control])
                y = federated_de[key]
                y_err = federated_de_std[key]
                ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[control], alpha=.1)

                all_key = key.replace('de', 'all')
                all_key_list = all_key.split('_')
                all_key_list[-2] = '0'
                all_key = '_'.join(all_key_list)

                control = federated_all_pic_key
                # ax_1.errorbar(x, federated_all[all_key], yerr=federated_all_std[all_key], color=color_dict[control], linestyle=linestyle_dict[control],
                #             label=control_dict[control])
                ax_1.errorbar(x, federated_all[all_key], yerr=None, color=color_dict[control], linestyle=linestyle_dict[control],
                            label=control_dict[control])
                y = federated_all[all_key]
                y_err = federated_all_std[all_key]
                ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[control], alpha=.1)
                # ax_1.fill_between(x, federated_de[key], federated_all[all_key])
                # ax_1.fill_between(x, (y - y_err), (y + y_err), color=color_dict[pivot], alpha=.1)

                # ax_1.set_xticks(x)
                # ax_1.set_xticks()
                ax_1.set_xlabel('Communication Rounds', fontsize=fontsize['label'])
                ax_1.set_ylabel(metric_name, fontsize=fontsize['label'])
                ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
                ax_1.legend(loc=label_loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.1')
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'lc', *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return




def make_vis_lc_best(df_exp, df_history):
    control_dict = {'Joint': 'Joint', 'Alone': 'Alone', 'MDR': 'MDR',
                    'constant-0.01_constant': 'DMTMDR', 'constant-0.05_constant': 'DMTMDR',
                    'constant-0.1_constant': 'DMTMDR', 'constant-0.3_constant': 'DMTMDR',
                    'constant-1_constant': 'DMTMDR', 'optim-0.1_constant': 'DMTMDR'}
    color_dict = {'Joint': 'blue', 'Alone': 'black', 'MDR': 'orange',
                  'constant-0.01_constant': 'red', 'constant-0.05_constant': 'red',
                  'constant-0.1_constant': 'red', 'constant-0.3_constant': 'red',
                  'constant-1_constant': 'red', 'optim-0.1_constant': 'red'}
    linestyle_dict = {'Joint': '-.', 'Alone': '--', 'MDR': ':',
                      'constant-0.01_constant': '-', 'constant-0.05_constant': '-',
                      'constant-0.1_constant': '-', 'constant-0.3_constant': '-',
                      'constant-1_constant': '-', 'optim-0.1_constant': '-'}
    marker_dict = {'Joint': 'X', 'Alone': 'x', 'MDR': 'p',
                   'constant-0.01_constant': 'd', 'constant-0.05_constant': 'd',
                   'constant-0.1_constant': 'd', 'constant-0.3_constant': 'd',
                   'constant-1_constant': 'd', 'optim-0.1_constant': 'd'}
    label_loc_dict = {'Loss': 'upper right', 'RMSE': 'upper right', 'NDCG': 'lower right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_history:
        df_name_list = df_name.split('_')
        info = df_name_list[3]
        metric_name = df_name_list[-2]
        stat = df_name_list[-1]
        valid_mask = info == '0' and metric_name in ['RMSE', 'NDCG'] and stat == 'mean'
        if valid_mask:
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            joint = {}
            alone = {}
            mdr = {}
            for (index, row) in df_exp[df_name].iterrows():
                index_list = index.split('_')
                if 'joint' in index_list and len(index_list) == 2:
                    joint_ = row.to_numpy()
                    joint[index] = joint_[~np.isnan(joint_)]
                if 'alone' in index_list and len(index_list) == 2:
                    alone_ = row.to_numpy()
                    alone[index] = alone_[~np.isnan(alone_)]
                if 'mdr' in index_list and len(index_list) == 2:
                    mdr_ = row.to_numpy()
                    mdr[index] = mdr_[~np.isnan(mdr_)]
            assist = {}
            for (index, row) in df_history[df_name].iterrows():
                index_list = index.split('_')
                if 'assist' in index_list and len(index_list) == 5 and index_list[-1] == '1' and \
                        index_list[-2] != 'optim':
                    assist_ = row.to_numpy()
                    assist[index] = assist_[~np.isnan(assist_)]
            joint_std = {}
            alone_std = {}
            mdr_std = {}
            for (index, row) in df_exp[df_name_std].iterrows():
                index_list = index.split('_')
                if 'joint' in index_list and len(index_list) == 2:
                    joint_ = row.to_numpy()
                    joint_std[index] = joint_[~np.isnan(joint_)]
                if 'alone' in index_list and len(index_list) == 2:
                    alone_ = row.to_numpy()
                    alone_std[index] = alone_[~np.isnan(alone_)]
                if 'mdr' in index_list and len(index_list) == 2:
                    mdr_ = row.to_numpy()
                    mdr_std[index] = mdr_[~np.isnan(mdr_)]
            assist_std = {}
            for (index, row) in df_history[df_name_std].iterrows():
                index_list = index.split('_')
                if 'assist' in index_list and len(index_list) == 5 and index_list[-1] == '1' and \
                        index_list[-2] != 'optim':
                    assist_ = row.to_numpy()
                    assist_std[index] = assist_[~np.isnan(assist_)]
            joint_values = np.array(list(joint.values())).reshape(-1)
            joint_std_values = np.array(list(joint_std.values())).reshape(-1)
            alone_values = np.array(list(alone.values())).reshape(-1)
            alone_std_values = np.array(list(alone_std.values())).reshape(-1)
            mdr_values = np.array(list(mdr.values()))
            mdr_std_values = np.array(list(mdr_std.values())).reshape(-1)
            assist_values = np.array(list(assist.values()))
            assist_std_values = np.array(list(assist_std.values()))
            assist_controls = list(assist.keys())
            if metric_name in ['NDCG']:
                joint_best_idx = np.argmax(joint_values)
                alone_best_idx = np.argmax(alone_values)
                mdr_best_idx = np.argmax(mdr_values)
                assist_best_idx = np.argmax(assist_values[:, -1].reshape(-1))
            else:
                joint_best_idx = np.argmin(joint_values)
                alone_best_idx = np.argmin(alone_values)
                mdr_best_idx = np.argmin(mdr_values)
                assist_best_idx = np.argmin(assist_values[:-1, -1].reshape(-1))
            x = np.arange(11)
            joint = joint_values[joint_best_idx]
            joint_std = joint_std_values[joint_best_idx]
            joint = np.full(x.shape, joint)
            joint_std = np.full(x.shape, joint_std)
            alone = alone_values[alone_best_idx]
            alone_std = alone_std_values[alone_best_idx]
            alone = np.full(x.shape, alone)
            alone_std = np.full(x.shape, alone_std)
            mdr = mdr_values[mdr_best_idx]
            mdr_std = mdr_std_values[mdr_best_idx]
            mdr = np.full(x.shape, mdr)
            mdr_std = np.full(x.shape, mdr_std)
            assist_control = '_'.join([*assist_controls[assist_best_idx].split('_')[2:4]])
            assist = assist_values[assist_best_idx]
            assist_std = assist_std_values[assist_best_idx]
            fig_name = '_'.join([*df_name_list[:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            control = 'Joint'
            ax_1.errorbar(x, joint, joint_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            control = 'MDR'
            ax_1.errorbar(x, mdr, mdr_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            control = 'Alone'
            ax_1.errorbar(x, alone, alone_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            control = assist_control
            ax_1.errorbar(x, assist, assist_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            ax_1.set_xticks(x)
            ax_1.set_xlabel('Assistance Rounds', fontsize=fontsize['label'])
            ax_1.set_ylabel(metric_name, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=label_loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'lc_best', *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_match(df_exp):
    control_dict = {'mdr': 'MDR', 'assist': 'DMTMDR'}
    color_dict = {'mdr': 'orange', 'assist': 'red'}
    linestyle_dict = {'mdr': ':', 'assist': '-'}
    marker_dict = {'mdr': 'p', 'assist': 'd'}
    label_loc_dict = {'Loss': 'upper right', 'RMSE': 'upper right', 'NDCG': 'lower right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_exp:
        df_name_list = df_name.split('_')
        data_mode = df_name_list[1]
        info = df_name_list[3]
        metric_name = df_name_list[-2]
        stat = df_name_list[-1]
        valid_mask = data_mode == 'user' and info == '0' and metric_name in ['RMSE', 'NDCG'] and stat == 'mean'
        if valid_mask:
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            alone = {}
            mdr = {}
            assist = {}
            for (index, row) in df_exp[df_name].iterrows():
                index_list = index.split('_')
                if 'alone' in index_list and len(index_list) == 2:
                    alone_ = row.to_numpy()
                    alone[index] = alone_[~np.isnan(alone_)]
                if 'mdr' in index_list and len(index_list) in [2, 5]:
                    mdr_ = row.to_numpy()
                    mdr[index] = mdr_[~np.isnan(mdr_)]
                if 'assist' in index_list and len(index_list) == 5 and \
                        index_list[-2] != 'optim':
                    assist_ = row.to_numpy()
                    assist[index] = assist_[~np.isnan(assist_)]
            alone_std = {}
            mdr_std = {}
            assist_std = {}
            for (index, row) in df_exp[df_name_std].iterrows():
                index_list = index.split('_')
                if 'alone' in index_list and len(index_list) == 2:
                    alone_ = row.to_numpy()
                    alone_std[index] = alone_[~np.isnan(alone_)]
                if 'mdr' in index_list and len(index_list) in [2, 5]:
                    mdr_ = row.to_numpy()
                    mdr_std[index] = mdr_[~np.isnan(mdr_)]
                if 'assist' in index_list and len(index_list) == 5 and \
                        index_list[-2] != 'optim':
                    assist_ = row.to_numpy()
                    assist_std[index] = assist_[~np.isnan(assist_)]
            alone_values = np.array(list(alone.values())).reshape(-1)
            alone_std_values = np.array(list(alone_std.values())).reshape(-1)
            mdr_values = np.array(list(mdr.values()))
            mdr_std_values = np.array(list(mdr_std.values())).reshape(-1)
            assist_ = {}
            assist_std_ = {}
            for k in assist:
                k_list = k.split('_')
                if k_list[2] == 'constant-1.0' or k_list[2] == 'constant-1':
                    k_list[2] = 'constant-1.0'
                    k_ = '_'.join(k_list)
                    assist_[k_] = assist[k]
                    assist_std_[k_] = assist_std[k]
                else:
                    assist_[k] = assist[k]
                    assist_std_[k] = assist_std[k]
            assist = assist_
            assist_std = assist_std_
            assist_values = np.array(list(assist.values()))
            assist_std_values = np.array(list(assist_std.values()))
            assist_controls = list(assist.keys())
            if metric_name in ['NDCG']:
                alone_best_idx = np.argmax(alone_values)
                mdr_best_idx = np.argmax(mdr_values[:3])
            else:
                alone_best_idx = np.argmin(alone_values)
                mdr_best_idx = np.argmin(mdr_values[:3])
            x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            alone = alone_values[alone_best_idx]
            alone_std = alone_std_values[alone_best_idx]
            mdr = mdr_values[3:].reshape(3, -1)[mdr_best_idx]
            mdr_std = mdr_std_values[3:].reshape(3, -1)[mdr_best_idx]
            mdr = [alone] + mdr.tolist() + [mdr_values[:3][mdr_best_idx].item()]
            mdr_std = [alone_std] + mdr_std.tolist() + [mdr_std_values[:3][mdr_best_idx].item()]
            assist_control = '_'.join(assist_controls[-1].split('_')[2:4])
            assist_values = [assist_values[i].item() for i in range(len(assist_values)) if
                             assist_control in assist_controls[i]]
            assist_std_values = [assist_std_values[i].item() for i in range(len(assist_std_values)) if
                                 assist_control in assist_controls[i]]
            assist = [alone] + assist_values[1:] + [assist_values[0]]
            assist_std = [alone_std] + assist_std_values[1:] + [assist_std_values[0]]
            fig_name = '_'.join([*df_name_list[:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            control = 'mdr'
            ax_1.errorbar(x, mdr, mdr_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            control = 'assist'
            ax_1.errorbar(x, assist, assist_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            ax_1.set_xlabel('Aligntment Ratio', fontsize=fontsize['label'])
            ax_1.set_ylabel(metric_name, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=label_loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'match', *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return


def make_vis_cs(df_exp, df_each):
    control_dict = {'alone': 'Alone', 'mdr': 'MDR', 'assist': 'DMTMDR'}
    color_dict = {'alone': 'black', 'mdr': 'orange', 'assist': 'red'}
    linestyle_dict = {'alone': '--', 'mdr': ':', 'assist': '-'}
    marker_dict = {'alone': 'x', 'mdr': 'p', 'assist': 'd'}
    label_loc_dict = {'Loss': 'upper right', 'RMSE': 'upper right', 'NDCG': 'lower right'}
    fontsize = {'legend': 16, 'label': 16, 'ticks': 16}
    figsize = (5, 4)
    fig = {}
    ax_dict_1 = {}
    for df_name in df_exp:
        df_name_list = df_name.split('_')
        data_mode = df_name_list[1]
        info = df_name_list[3]
        metric_name = df_name_list[-2]
        stat = df_name_list[-1]
        valid_mask = data_mode == 'user' and info == '0' and metric_name in ['RMSE', 'NDCG'] and stat == 'mean'
        if valid_mask:
            df_name_std = '_'.join([*df_name_list[:-1], 'std'])
            alone = {}
            mdr = {}
            assist = {}
            for (index, row) in df_exp[df_name].iterrows():
                index_list = index.split('_')
                if 'alone' in index_list and len(index_list) == 7:
                    alone_ = row.to_numpy()
                    alone[index] = alone_[~np.isnan(alone_)]
                if 'mdr' in index_list and len(index_list) == 7:
                    mdr_ = row.to_numpy()
                    mdr[index] = mdr_[~np.isnan(mdr_)]
                if 'assist' in index_list and len(index_list) == 7 and \
                        index_list[-2] != 'optim':
                    assist_ = row.to_numpy()
                    assist[index] = assist_[~np.isnan(assist_)]
            for (index, row) in df_each[df_name].iterrows():
                index_list = index.split('_')
                if 'alone' in index_list and ('base' in index_list or 'ae' in index_list) and len(index_list) == 2:
                    alone_ = row.to_numpy()
                    alone[index] = np.array([alone_[~np.isnan(alone_)][0]])
                if 'mdr' in index_list and len(index_list) == 2:
                    mdr_ = row.to_numpy()
                    mdr[index] = np.array([mdr_[~np.isnan(mdr_)][0]])
                if 'assist' in index_list and len(index_list) == 5 and index_list[-1] == '1' and \
                        index_list[-2] != 'optim':
                    assist_ = row.to_numpy()
                    assist[index] = np.array([assist_[~np.isnan(assist_)][0]])
            alone_std = {}
            mdr_std = {}
            assist_std = {}
            for (index, row) in df_exp[df_name_std].iterrows():
                index_list = index.split('_')
                if 'alone' in index_list and len(index_list) == 7:
                    alone_ = row.to_numpy()
                    alone_std[index] = alone_[~np.isnan(alone_)]
                if 'mdr' in index_list and len(index_list) == 7:
                    mdr_ = row.to_numpy()
                    mdr_std[index] = mdr_[~np.isnan(mdr_)]
                if 'assist' in index_list and len(index_list) == 7:
                    assist_ = row.to_numpy()
                    assist_std[index] = assist_[~np.isnan(assist_)]
            for (index, row) in df_each[df_name_std].iterrows():
                index_list = index.split('_')
                if 'alone' in index_list and ('base' in index_list or 'ae' in index_list) and len(index_list) == 2:
                    alone_ = row.to_numpy()
                    alone_std[index] = np.array([alone_[~np.isnan(alone_)][0]])
                if 'mdr' in index_list and len(index_list) == 2:
                    mdr_ = row.to_numpy()
                    mdr_std[index] = np.array([mdr_[~np.isnan(mdr_)][0]])
                if 'assist' in index_list and len(index_list) == 5 and index_list[-1] == '1':
                    assist_ = row.to_numpy()
                    assist_std[index] = np.array([assist_[~np.isnan(assist_)][0]])
            alone_values = np.array(list(alone.values())).reshape(-1)
            alone_std_values = np.array(list(alone_std.values())).reshape(-1)
            mdr_values = np.array(list(mdr.values()))
            mdr_std_values = np.array(list(mdr_std.values())).reshape(-1)
            assist_ = {}
            assist_std_ = {}
            for k in assist:
                k_list = k.split('_')
                if k_list[2] == 'constant-1.0' or k_list[2] == 'constant-1':
                    k_list[2] = 'constant-1.0'
                    k_ = '_'.join(k_list)
                    assist_[k_] = assist[k]
                    assist_std_[k_] = assist_std[k]
                else:
                    assist_[k] = assist[k]
                    assist_std_[k] = assist_std[k]
            assist = assist_
            assist_std = assist_std_
            assist_values = np.array(list(assist.values()))
            assist_std_values = np.array(list(assist_std.values()))
            assist_controls = list(assist.keys())
            if metric_name in ['NDCG']:
                alone_best_idx = np.argmax(alone_values[-2:])
                mdr_best_idx = np.argmax(mdr_values[-3:])
            else:
                alone_best_idx = np.argmin(alone_values[-2:])
                mdr_best_idx = np.argmin(mdr_values[-3:])
            x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            alone = alone_values[:-2].reshape(2, -1)[alone_best_idx]
            alone_std = alone_std_values[:-2].reshape(2, -1)[alone_best_idx]
            alone = alone.tolist() + [alone_values[-2:][alone_best_idx].item()]
            alone_std = alone_std.tolist() + [alone_std_values[-2:][alone_best_idx].item()]
            mdr = mdr_values[:-3].reshape(3, -1)[mdr_best_idx]
            mdr_std = mdr_std_values[:-3].reshape(3, -1)[mdr_best_idx]
            mdr = mdr.tolist() + [mdr_values[-3:][mdr_best_idx].item()]
            mdr_std = mdr_std.tolist() + [mdr_std_values[-3:][mdr_best_idx].item()]
            assist_control = '_'.join(assist_controls[0].split('_')[2:4])
            assist_values = [assist_values[i].item() for i in range(len(assist_values)) if
                             assist_control in assist_controls[i]]
            assist_std_values = [assist_std_values[i].item() for i in range(len(assist_std_values)) if
                                 assist_control in assist_controls[i]]
            assist = assist_values
            assist_std = assist_std_values
            fig_name = '_'.join([*df_name_list[:-1]])
            fig[fig_name] = plt.figure(fig_name, figsize=figsize)
            if fig_name not in ax_dict_1:
                ax_dict_1[fig_name] = fig[fig_name].add_subplot(111)
            ax_1 = ax_dict_1[fig_name]
            control = 'alone'
            ax_1.errorbar(x, alone, alone_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            control = 'mdr'
            ax_1.errorbar(x, mdr, mdr_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            control = 'assist'
            ax_1.errorbar(x, assist, assist_std,
                          color=color_dict[control], linestyle=linestyle_dict[control],
                          label=control_dict[control], marker=marker_dict[control])
            ax_1.set_xlabel('Cold Start Ratio', fontsize=fontsize['label'])
            ax_1.set_ylabel(metric_name, fontsize=fontsize['label'])
            ax_1.xaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.yaxis.set_tick_params(labelsize=fontsize['ticks'])
            ax_1.legend(loc=label_loc_dict[metric_name], fontsize=fontsize['legend'])
    for fig_name in fig:
        fig[fig_name] = plt.figure(fig_name)
        ax_dict_1[fig_name].grid(linestyle='--', linewidth='0.5')
        fig[fig_name].tight_layout()
        control = fig_name.split('_')
        dir_path = os.path.join(vis_path, 'cs', *control[:-1])
        fig_path = os.path.join(dir_path, '{}.{}'.format(fig_name, save_format))
        makedir_exist_ok(dir_path)
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        plt.close(fig_name)
    return

if __name__ == '__main__':
    main()
