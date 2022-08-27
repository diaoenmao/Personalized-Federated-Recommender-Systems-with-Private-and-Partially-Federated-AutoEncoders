#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
#!/bin/bash
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --control_name cuda --device ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
#!/bin/bash
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
#!/bin/bash
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_l_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_l_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_l_max_1_l
wait
#!/bin/bash
#!/bin/bash
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
#!/bin/bash
#!/bin/bash
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_1_iid_g_max_0_l
wait
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_all_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_all_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="2" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_all.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_1_iid_g_max_0_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python train_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML100K_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name ML1M_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_ex_fedavg_de_ae_1_iid_g_max_1_l
wait
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_0_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_0_iid_g_max_1_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_1_iid_g_max_0_l&
CUDA_VISIBLE_DEVICES="0" python test_privacy_federated_decoder.py --init_seed 0 --world_size 1 --num_experiments 1 --resume_mode 0 --log_interval 0.25 --device cuda --control_name Douban_user_im_fedavg_de_ae_1_iid_g_max_1_l
wait
