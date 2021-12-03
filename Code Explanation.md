Command:

cuda => cpu if dont have gpu

1. Example

   1. python train_recsys_joint.py --control_name ML100K_user_implicit_ae_random-2_constant-0.1_constant

2. Process:

   1. config.py / process_args

      1. 处理args，转为字典结构

   2. utils.py / process_control

      1. 拆分cfg，赋予模型参数

         

Train:





Test:





Whole Flow ():

1. 处理command (参照above command)
2. train_recsys_joint.py / runExperiment() (train_recsys_joint.py as example)
3. Init dataset
   1. Data.py / fetch_dataset()
      1. 初始化datasets中的class (eval()执行string, 完成初始化)
      2. 

fetch_dataset => import datasets => init ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP'] if not exists => 

















Data loading:





