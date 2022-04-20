#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_ex_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_ex_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_ex_fedavg_de_ae_0_iid_g_300_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_ex_fedavg_de_ae_0_iid_g_300_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_im_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_im_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_im_fedavg_de_ae_0_iid_g_300_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML10M_user_im_fedavg_de_ae_0_iid_g_300_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_ex_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_ex_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_ex_fedavg_de_ae_0_iid_g_300_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_ex_fedavg_de_ae_0_iid_g_300_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_im_fedavg_de_ae_0_iid_g_100_0_l
wait
CUDA_VISIBLE_DEVICES="1" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_im_fedavg_de_ae_0_iid_g_100_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_im_fedavg_de_ae_0_iid_g_300_0_l
wait
CUDA_VISIBLE_DEVICES="3" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name ML20M_user_im_fedavg_de_ae_0_iid_g_300_1_l
wait
