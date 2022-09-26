#!/bin/bash

# run joint
python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file joint --data ML1M_Anime
python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file joint --data ML1M_Anime

# # federate => average all
python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file fedAvg --data ML1M_Anime
python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file fedAvg --data ML1M_Anime

# # federate => average decoder
python3 large_scale_create_command.py --run train --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file personalFR --data ML1M_Anime
python3 large_scale_create_command.py --run test --device cuda --num_gpus 4 --round 32 --world_size 1 --num_experiments 4 --experiment_step 1 --init_seed 0 --resume_mode 0 --log_interval 0.25 --file personalFR --data ML1M_Anime

# # reassign max commands
python3 large_scale_reassign_command.py

