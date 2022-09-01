#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="1" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_joint_NA_ae_1_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="3" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_joint_NA_ae_1_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="1" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_joint_NA_ae_1_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="3" python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_joint_NA_ae_1_iid_g_1_0_l&
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
