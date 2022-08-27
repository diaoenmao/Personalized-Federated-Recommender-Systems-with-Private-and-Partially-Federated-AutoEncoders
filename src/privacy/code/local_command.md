python train_privacy_joint.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cpu --control_name ML100K_user_ex_joint_NA_ae_0_iid_g_1_0_l

python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cpu --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_1_0_l

python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cpu --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_100_0_l

python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cpu --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_100_0_l

python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cpu --control_name Douban_user_ex_fedavg_all_ae_0_iid_g_100_0_l