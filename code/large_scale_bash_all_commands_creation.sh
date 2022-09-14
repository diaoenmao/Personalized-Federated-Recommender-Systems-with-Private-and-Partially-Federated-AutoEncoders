#!/bin/bash

# # run joint
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M_Douban
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data ML100K_ML1M_ML10M_ML20M_Douban

# # federate => average all
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML100K_ML1M_ML10M_ML20M_Douban
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML100K_ML1M_ML10M_ML20M_Douban

# # federate => average decoder
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M_Douban
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML100K_ML1M_ML10M_ML20M_Douban



# # federate => average all
# python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML1M
# python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data ML1M




# run joint
python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data Netflix
python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data Netflix

# # federate => average all
python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data Netflix
python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data Netflix

# # federate => average decoder
python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data Netflix
python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data Netflix

# # reassign max commands
# python3 large_scale_reassign_command.py

# # federate => average decoder
# python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML1M
# python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data ML1M

# # reassign max commands
# python3 large_scale_reassign_command.py

# run joint
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data Douban
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_joint --data Douban

# # federate => average all
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data Douban
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_all --data Douban

# # federate => average decoder
# python3 large_scale_commands_creation.py --run train --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data Douban
# python3 large_scale_commands_creation.py --run test --device cuda --num_gpus 4 --round 4 --world_size 1 --num_experiments 1 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file privacy_federated_decoder --data Douban