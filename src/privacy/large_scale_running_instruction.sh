python large_scale_data_preparation.py #to prepare data

bash large_scale_train_privacy_federated_all.sh; bash large_scale_test_privacy_federated_all.sh #on the 1st server

bash large_scale_train_privacy_federated_decoder.sh;  bash large_scale_test_privacy_federated_decoder.sh #on the 2nd server

bash large_scale_train_privacy_joint.sh; bash large_scale_test_privacy_joint.sh #on the 3rd server

# For results: 
# 1. download output/run folder (for tensorboard viewing)
# 2. download output/result folder (info saved from test stage)
