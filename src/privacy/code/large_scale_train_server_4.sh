#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l
wait