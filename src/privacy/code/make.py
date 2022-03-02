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
parser.add_argument('--log_interval', default=None, type=float)
args = vars(parser.parse_args())


def make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval, control_name):
    control_names = []
    for i in range(len(control_name)):
        control_names.extend(list('_'.join(x) for x in itertools.product(*control_name[i])))
    control_names = [control_names]
    controls = script_name + init_seeds + world_size + num_experiments + resume_mode + log_interval + control_names
    controls = list(itertools.product(*controls))
    return controls

'''
run: train or test
init_seed: 0
world_size: 1
num_experiments: 1
resume_mode: 0
log_interval: 0.25
num_gpus: 12
round: 1
experiment_step: 1
file: train_后面的，例如privacy_joint
data: ML100K_ML1M_ML10M_ML20M

python make.py --run train --num_gpus 12 --round 1 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M
'''

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
    file = args['file']
    data = args['data'].split('_')
    

    gpu_ids = [','.join(str(i) for i in list(range(x, x + world_size))) for x in list(range(0, num_gpus, world_size))]
    init_seeds = [list(range(init_seed, init_seed + num_experiments, experiment_step))]
    world_size = [[world_size]]
    num_experiments = [[experiment_step]]
    resume_mode = [[resume_mode]]
    log_interval = [[log_interval]]
    filename = '{}_{}'.format(run, file)

    if file == 'privacy_joint':
        controls = []
        script_name = [['{}_privacy_joint.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['joint'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['joint'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['joint'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['joint'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml20m_controls)

    elif file == 'privacy_fedsgd':
        controls = []
        script_name = [['{}_privacy_fedsgd.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['fedsgd'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['fedsgd'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['fedsgd'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['fedsgd'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['large']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml20m_controls)
    elif file == 'privacy_federated_all':
        controls = []
        script_name = [['{}_privacy_federated_all.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['100'], ['0'], ['large']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['100'], ['0'], ['large']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['100'], ['0'], ['large']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['100'], ['0'], ['large']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml20m_controls)
    elif file == 'privacy_federated_decoder':
        controls = []
        script_name = [['{}_privacy_federated_decoder.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['decoder'], ['ae'],
                             ['0'], ['iid'], ['100', 'max'], ['0'], ['large']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['100', 'max'], ['0'], ['large']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['100'], ['0'], ['large']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['explicit', 'implicit'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['100'], ['0'], ['large']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml20m_controls)
    else:
        raise ValueError('Not valid file')

    print('controls', controls)
    s = '#!/bin/bash\n'
    k = 0
    for i in range(len(controls)):
        controls[i] = list(controls[i])
        s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --init_seed {} --world_size {} --num_experiments {} ' \
                '--resume_mode {} --log_interval {} --control_name {}&\n'.format(gpu_ids[k % len(gpu_ids)], *controls[i])
        if k % round == round - 1:
            s = s[:-2] + '\nwait\n'
        k = k + 1
    if s[-5:-1] != 'wait':
        s = s + 'wait\n'
    print(s)
    run_file = open('./{}.sh'.format(filename), 'w')
    run_file.write(s)
    run_file.close()
    return


if __name__ == '__main__':
    main()
