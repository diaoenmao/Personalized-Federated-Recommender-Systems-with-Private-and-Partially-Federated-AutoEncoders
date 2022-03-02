#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_explicit_joint_ae_0_iid
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_implicit_joint_ae_0_iid
wait
