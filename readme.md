

# Personalized Federated Recommender Systems with Private and Partially Federated AutoEncoders

This is an implementation of Personalized Federated Recommender Systems with Private and Partially Federated AutoEncoders

![image-20220921204258697](/Users/qile/Library/Application Support/typora-user-images/image-20220921204258697.png)



### Requirements

-------

- see requirements.txt



### Instruction

-----

- Global hyperparameters are configured in config.yml
- Hyperparameters can be found at process_control() in utils.py
- fed.py contrains aggregation and separation of clients



### Examples

-----

- Train ML1M dataset (IID) with AE model, Joint, 1 client, explicit feedback, Compress Transmission(False)

  ```ruby
  python train_joint.py --control_name ML1M_user_im_joint_NA_ae_iid_g_1_0_l
  ```

- Train Anime dataset (IID) with AE model, FedAvg, 100 clients, explicit feedback, Compress Transmission(False)

  ```ruby
  python train_fedAvg.py --control_name Anime_user_ex_fedavg_FedAvg_ae_iid_100_0_l
  ```

- Test ML1M dataset (IID) with AE model, PersonalFR, 6040 client(1 user/client), implicit feedback, Compress Transmission(True)

  ```ruby
  python test_personalFR.py --control_name ML1M_user_im_fedavg_PersonalFR_ae_iid_max_1_l
  ```





### Results

------

![image-20220925183102709](/Users/qile/Library/Application Support/typora-user-images/image-20220925183102709.png)

![image-20220925183123591](/Users/qile/Library/Application Support/typora-user-images/image-20220925183123591.png)

![image-20220921205314472](/Users/qile/Library/Application Support/typora-user-images/image-20220921205314472.png)



![image-20220921205338411](/Users/qile/Library/Application Support/typora-user-images/image-20220921205338411.png)



![image-20220921205423416](/Users/qile/Library/Application Support/typora-user-images/image-20220921205423416.png)



### Acknowledgement

-----

*Qi Le*

*Enmao Diao*

*Xinran Wang*

*Ali Anwar*

*Vahid Tarokh*

*Jie Ding*