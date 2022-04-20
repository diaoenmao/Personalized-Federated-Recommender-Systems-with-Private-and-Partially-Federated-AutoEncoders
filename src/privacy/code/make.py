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
file: train_后面的, 例如privacy_joint
data: ML100K_ML1M_ML10M_ML20M

python make.py --run train --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M

control:
  data_name: ML100K (Name of the dataset)
  data_mode: user (user or item)
  target_mode: ex (explicit(ex) or implicit(im))
  train_mode: fedavg (joint, fedsgd, fedavg)
  federated_mode: de (all or decoder(de))
  model_name: ae 
  info: 1 (1: use user attribute, 0: not use user attribute)
  data_split_mode: 'iid' (iid or non-iid)
  update_best_model: 'g' (global(g) or local(l))
  num_nodes: 100 (1, 100, 300, max)
  compress_transmission: 1 (1: compress, 0: not compress)
  experiment_size: 'l' (l(l): transfer parameters to cpu)
  
# experiment
fine_tune: 0
fine_tune_lr: 0.1
fine_tune_batch_size: 5
fine_tune_epoch: 5
fix_layers: last
fine_tune_scheduler: CosineAnnealingLR
num_workers: 0
init_seed: 0
num_experiments: 1
log_interval: 0.25
device: cuda
resume_mode: 0
verbose: False

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
            control_name = [[['ML100K'], ['user'], ['ex', 'im'], ['joint'], ['NA'], ['ae'],
                             ['0'], ['iid'], ['NA'], ['1'], ['0'], ['l']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['joint'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['l']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['ex', 'im'], ['joint'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['l']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['ex', 'im'], ['joint'], ['None'], ['ae'],
                             ['0'], ['iid'], ['1'], ['0'], ['l']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml20m_controls)

    # elif file == 'privacy_fedsgd':
    #     controls = []
    #     script_name = [['{}_privacy_fedsgd.py'.format(run)]]
    #     if 'ML100K' in data:
    #         control_name = [[['ML100K'], ['user'], ['ex', 'im'], ['fedsgd'], ['None'], ['ae'],
    #                          ['0'], ['iid'], ['1'], ['0'], ['l']]]
    #         ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
    #                                         control_name)
    #         controls.extend(ml100k_controls)
    #     if 'ML1M' in data:
    #         control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['fedsgd'], ['None'], ['ae'],
    #                          ['0'], ['iid'], ['1'], ['0'], ['l']]]
    #         ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
    #                                       control_name)
    #         controls.extend(ml1m_controls)
    #     if 'ML10M' in data:
    #         control_name = [[['ML10M'], ['user'], ['ex', 'im'], ['fedsgd'], ['None'], ['ae'],
    #                          ['0'], ['iid'], ['1'], ['0'], ['l']]]
    #         ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
    #                                        control_name)
    #         controls.extend(ml10m_controls)
    #     if 'ML20M' in data:
    #         control_name = [[['ML20M'], ['user'], ['ex', 'im'], ['fedsgd'], ['None'], ['ae'],
    #                          ['0'], ['iid'], ['1'], ['0'], ['l']]]
    #         ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
    #                                        control_name)
    #         controls.extend(ml20m_controls)
    elif file == 'privacy_federated_all':
        controls = []
        script_name = [['{}_privacy_federated_all.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['ex', 'im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', 'max'], ['0'], ['l']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', 'max'], ['0'], ['l']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['ex', 'im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', '300'], ['0'], ['l']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['ex', 'im'], ['fedavg'], ['all'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', '300'], ['0'], ['l']]]
            ml20m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml20m_controls)
    elif file == 'privacy_federated_decoder':
        controls = []
        script_name = [['{}_privacy_federated_decoder.py'.format(run)]]
        if 'ML100K' in data:
            control_name = [[['ML100K'], ['user'], ['ex', 'im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', 'max'], ['0', '1'], ['l']]]
            ml100k_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                            control_name)
            controls.extend(ml100k_controls)
        if 'ML1M' in data:
            control_name = [[['ML1M'], ['user'], ['ex', 'im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', 'max'], ['0', '1'], ['l']]]
            ml1m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                          control_name)
            controls.extend(ml1m_controls)
        if 'ML10M' in data:
            control_name = [[['ML10M'], ['user'], ['ex', 'im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', '300'], ['0', '1'], ['l']]]
            ml10m_controls = make_controls(script_name, init_seeds, world_size, num_experiments, resume_mode, log_interval,
                                           control_name)
            controls.extend(ml10m_controls)
        if 'ML20M' in data:
            control_name = [[['ML20M'], ['user'], ['ex', 'im'], ['fedavg'], ['de'], ['ae'],
                             ['0'], ['iid'], ['g'], ['100', '300'], ['0', '1'], ['l']]]
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
