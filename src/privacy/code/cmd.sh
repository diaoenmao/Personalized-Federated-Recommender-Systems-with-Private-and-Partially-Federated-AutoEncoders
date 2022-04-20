#!/bin/bash

# run joint
python3 make.py --run train --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M
python3 make.py --run test --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M

# federate => average all
python3 make.py --run train --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML100K_ML1M_ML10M_ML20M
python3 make.py --run test --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML100K_ML1M_ML10M_ML20M

# federate => average decoder
python3 make.py --run train --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M
python3 make.py --run test --num_gpus 4 --round 1 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M