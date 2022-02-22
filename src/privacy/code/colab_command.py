!python train_privacy_joint.py --control_name ML1M_user_explicit_ae_0_iid_1 --train_mode not_private --federated_mode decoder --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cuda --resume_mode 0 --verbose False
!python test_privacy_joint.py --control_name ML1M_user_explicit_ae_0_iid_1 --train_mode not_private --federated_mode decoder --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cuda --resume_mode 0 --verbose False

!python train_privacy_federated.py --control_name ML1M_user_explicit_ae_0_iid_100_800_5 --train_mode private --federated_mode all --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cuda --resume_mode 0 --verbose False
!python test_privacy_federated.py --control_name ML1M_user_explicit_ae_0_iid_100_800_5 --train_mode private --federated_mode all --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cuda --resume_mode 0 --verbose False