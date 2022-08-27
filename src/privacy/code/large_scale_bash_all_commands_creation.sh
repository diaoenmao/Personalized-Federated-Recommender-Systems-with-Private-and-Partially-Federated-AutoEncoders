#!/bin/bash

# run joint
python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M_Douban
python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M_Douban

# federate => average all
python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML100K_ML1M_ML10M_ML20M_Douban
python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML100K_ML1M_ML10M_ML20M_Douban

# federate => average decoder
python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M_Douban
python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M_Douban