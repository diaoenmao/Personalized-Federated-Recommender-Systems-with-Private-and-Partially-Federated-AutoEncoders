# Personalized Federated Recommender Systems with Private and Partially Federated AutoEncoders

This is an implementation of [Personalized Federated Recommender Systems with Private and Partially Federated AutoEncoders](https://arxiv.org/abs/2212.08779)

![PersonalFR](/asset/PersonalFR.png)



## Requirements

- see requirements.txt

  

## Instruction

- Global hyperparameters are configured in config.yml

- Hyperparameters can be found at process_control() in utils.py

- fed.py contrains aggregation and separation of clients

  

## Examples

- Train ML1M dataset (IID) with AE model, Joint, 1 client, explicit feedback, Compress Transmission(False)

  ```ruby
  python train_joint.py --control_name ML1M_user_im_joint_NA_ae_iid_1_0_l
  ```

- Train Anime dataset (IID) with AE model, FedAvg, 100 clients, explicit feedback, Compress Transmission(False)

  ```ruby
  python train_fedAvg.py --control_name Anime_user_ex_fedavg_FedAvg_ae_iid_100_0_l
  ```

- Test ML1M dataset (IID) with AE model, PersonalFR, 6040 client(1 user/client), implicit feedback, Compress Transmission(True)

  ```ruby
  python test_personalFR.py --control_name ML1M_user_im_fedavg_PersonalFR_ae_iid_max_1_l
  ```



## Results

![table1](/asset/table1.png)

![table2](/asset/table2.png)

![figure1](/asset/figure1.png)

![figure2](/asset/figure2.png)

![figure3](/asset/figure3.png)



## Acknowledgement

*Qi Le*

*Enmao Diao*

*Xinran Wang*

*Ali Anwar*

*Vahid Tarokh*

*Jie Ding*
