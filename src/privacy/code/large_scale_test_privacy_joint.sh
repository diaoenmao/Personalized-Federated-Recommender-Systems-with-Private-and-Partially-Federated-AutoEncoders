#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="1" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_joint_NA_ae_1_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="3" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_joint_NA_ae_1_iid_g_1_0_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="1" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_joint_NA_ae_1_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="3" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_joint_NA_ae_1_iid_g_1_0_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML10M_user_ex_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="1" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML10M_user_ex_joint_NA_ae_1_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML10M_user_im_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="3" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML10M_user_im_joint_NA_ae_1_iid_g_1_0_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML20M_user_ex_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="1" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML20M_user_ex_joint_NA_ae_1_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML20M_user_im_joint_NA_ae_0_iid_g_1_0_l&
CUDA_VISIBLE_DEVICES="3" python test_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML20M_user_im_joint_NA_ae_1_iid_g_1_0_l
wait
