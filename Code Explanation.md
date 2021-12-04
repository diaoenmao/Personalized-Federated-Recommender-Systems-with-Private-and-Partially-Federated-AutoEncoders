**Privacy-Preserving Multi-Target Multi-Domain Recommender Systems with Assisted AutoEncoders:**



**Detailed Work Flow:**

Take **train_recsys_joint.py** as example

1. Command:
   1. python train_recsys_joint.py --control_name ML100K_user_implicit_ae_1_random-2_constant-0.1_constant_1 --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cpu --world_size 1 --resume_mode 0 --verbose False
   2. 修改device: cuda 为 device: cpu, 如果没有gpu
2. 创建解析器
   1. 创建
   2. **add_argument()** 加入参数，parser的值为yaml中的值。
   3. args = vars() 返回对象object的属性和属性值的字典对象, args此时为字典, args的值都用户输入的值，除了args['control']为yaml值。
   4. 传入process_args(args)处理, 更新cfg

3. Main():

   1. Utils.py / process_control()
      1. 处理模型参数
      2. 拆除cfg['control']
   2. 加上Init_seen, 跑num_experiments次runExperiment()

4. runExperiment():

   1. 

   





Train:





Test:





Technology Issue:

exec: 内置语句，执行储存在字符串或文件中的Python语句

Vars: 内置函数，返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值 类似 locals()。

















Data loading:













HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients:

