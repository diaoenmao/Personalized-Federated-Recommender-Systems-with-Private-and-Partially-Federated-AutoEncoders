import copy
import argparse
import itertools

parser = argparse.ArgumentParser(description='config')
parser.add_argument('--run', default='train', type=str)
parser.add_argument('--num_gpus', default=4, type=int)
parser.add_argument('--world_size', default=1, type=int)
parser.add_argument('--init_seed', default=0, type=int)
parser.add_argument('--round', default=4, type=int)
parser.add_argument('--experiment_step', default=1, type=int)
parser.add_argument('--num_experiments', default=1, type=int)
parser.add_argument('--resume_mode', default=0, type=int)
parser.add_argument('--file', default=None, type=str)
parser.add_argument('--data', default=None, type=str)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--log_interval', default=None, type=float)
args = vars(parser.parse_args())

def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, device, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + log_interval + device + control_names 
    controls = list(itertools.product(*controls))
    return controls

def main():
    run = args['run']
    num_gpus = args['num_gpus']
    round = args['round']
    world_size = args['world_size']
    experiment_step = args['experiment_step']
    init_seed = args['init_seed']
    num_experiments = args['num_experiments']
    resume_mode = args['resume_mode']
    log_interval = args['log_interval']
    device = args['device']
    file = args['file']
    data = args['data'].split('_')
    
    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    log_interval = [[log_interval]]
    device = [[device]]
    filename = '{}_{}'.format(run, file)
    
    if file == 'joint':
        controls = []
        script_name = [['{}_joint.py'.format(run)]]
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['iid'], ['1'], ['0'], ['l']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            device, control_name)
            controls.extend(ml1m_controls)
        if 'Anime' in data:
            control_name = [[['Anime'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['iid'], ['1'], ['0'], ['l']]]
            anime_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            device, control_name)
            controls.extend(anime_controls)
    elif file == 'fedAvg':
        controls = []
        script_name = [['{}_fedAvg.py'.format(run)]]
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex','im'], ['fedavg'], ['FedAvg'], ['ae'],
                             ['iid'], ['100', '300', 'max'], ['0'], ['l']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            device, control_name)
            controls.extend(ml1m_controls)
        if 'Anime' in data:
            control_name = [[['Anime'], ['user'], ['ex','im'], ['fedavg'], ['FedAvg'], ['ae'],
                             ['iid'], ['100','300'], ['0'], ['l']]]
            anime_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            device, control_name)
            controls.extend(anime_controls)
    elif file == 'personalFR':
        controls = []
        script_name = [['{}_personalFR.py'.format(run)]]
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex'], ['fedavg'], ['PersonalFR'], ['ae'],
                             ['iid'], ['100', '300', 'max'], ['1'], ['l']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            device, control_name)
            controls.extend(ml1m_controls)
        if 'Anime' in data:
            control_name = [[['Anime'], ['user'], ['ex','im'], ['fedavg'], ['PersonalFR'], ['ae'],
                             ['iid'], ['100','300'], ['1'], ['l']]]
            anime_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            device, control_name)
            controls.extend(anime_controls)
    else:
        raise ValueError('Not valid file')

    print('%$%$$controls', controls)
    s = '#!/bin/bash\n'
    k = 0

    s_for_max = ''
    k_for_max = 0
    k_round_for_max = 4
    max_controls = []
    
    new_controls = []
    for i in range(len(controls)):
        if 'max' in controls[i][-1]:
            max_controls.append(controls[i])
        else:
            new_controls.append(controls[i])
    
    controls = copy.deepcopy(new_controls)
    for i in range(len(controls)):
        # average computing time
        if i % 4 == 3:
            temp = controls[i-1]
            controls[i-1] = controls[i]
            controls[i] = temp

    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
                '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1

    for i in range(len(max_controls)):
        s_for_max = s_for_max + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
            '--resume_mode {} --log_interval {} --device {} --control_name {}&\n'.format(gpu_ids[k_for_max % len(gpu_ids)], *max_controls[i])

        if k_for_max % k_round_for_max == k_round_for_max - 1:
            s_for_max = s_for_max[:-2] + '\nwait\n'
        k_for_max = k_for_max + 1

    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    
    if len(s_for_max) >= 1 and s_for_max[-5:-1] != 'wait':
        s_for_max = s_for_max + 'wait\n'

    if run == 'train':
        filename = 'train_server_1'
    elif run == 'test':
        filename = 'test_server_1'
        
    run_file = open('./{}.sh'.format(f'large_scale_{filename}'), 'a')
    run_file.write(s)
    run_file.close()

    run_file = open('./{}.txt'.format(f'large_scale_{run}'), 'a')
    run_file.write(s_for_max)
    run_file.close()

    for i in range(4, 4):
        run_file = open('./{}.sh'.format(f'large_scale_train_server_{i}'), 'a')
        run_file.write('#!/bin/bash\n')
        run_file.close()

        run_file = open('./{}.sh'.format(f'large_scale_test_server_{i}'), 'a')
        run_file.write('#!/bin/bash\n')
        run_file.close()

    new_s = s.replace('CUDA_VISIBLE_DEVICES="0" ', '!')
    new_s = new_s.replace('CUDA_VISIBLE_DEVICES="1" ', '!')
    new_s = new_s.replace('CUDA_VISIBLE_DEVICES="2" ', '!')
    new_s = new_s.replace('CUDA_VISIBLE_DEVICES="3" ', '!')
    run_file = open('./{}.sh'.format(f'pre_run_large_scale_{filename}'), 'a')
    run_file.write(new_s)
    run_file.close()

    new_s_for_max = s_for_max.replace('CUDA_VISIBLE_DEVICES="0" ', '!')
    new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="1" ', '!')
    new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="2" ', '!')
    new_s_for_max = new_s_for_max.replace('CUDA_VISIBLE_DEVICES="3" ', '!')
    run_file = open('./{}.txt'.format(f'pre_run_large_scale_{run}'), 'a')
    run_file.write(new_s_for_max)
    run_file.close()

    for i in range(4, 4):
        run_file = open('./{}.sh'.format(f'pre_run_large_scale_train_server_{i}'), 'a')
        run_file.write('#!/bin/bash\n')
        run_file.close()

        run_file = open('./{}.sh'.format(f'pre_run_large_scale_test_server_{i}'), 'a')
        run_file.write('#!/bin/bash\n')
        run_file.close()

    return


if __name__ == '__main__':
    main()
