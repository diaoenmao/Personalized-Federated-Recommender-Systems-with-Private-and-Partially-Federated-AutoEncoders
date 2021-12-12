**Privacy-Preserving Multi-Target Multi-Domain Recommender Systems with Assisted AutoEncoders:**

yml:

​    \# control

​    control:

​        data_name: ML100K

​        data_mode: user

​        target_mode: implicit

​        model_name: ae #ae is autoencoder

​        info: 1

​        data_split_mode: random-2

​        ar: constant-0.1

​        aw: constant

​        match_rate: 1

​    \# experiment

​    num_workers: 0

​    init_seed: 0

​    num_experiments: 1

​    log_interval: 0.25

​    device: cuda

​    world_size: 1

​    resume_mode: 0

​    verbose: False

**Detailed Work Flow:**

Take **train_recsys_joint.py** as example

1. **Command:**
   
   1. python train_recsys_joint.py --control_name ML100K_user_implicit_ae_1_random-2_constant-0.1_constant_1 --num_workers 0 --init_seed 0 --num_experiments 1 --log_interval 0.25 --device cpu --world_size 1 --resume_mode 0 --verbose False
   2. 修改device: cuda 为 device: cpu, 如果没有gpu
   
2. **创建解析器**
   
   1. 创建
   2. ```add_argument()``` 加入参数，parser的值为yaml中的值。
   3. args = vars() 返回对象object的属性和属性值的字典对象, args此时为字典, args的值都用户输入的值，除了args['control']为yaml值。
   4. 传入process_args(args)处理, 更新cfg
   
3. **Main():**
   
   1. Utils.py / ```process_control()```
      1. 加入模型参数
      2. 拆除cfg['control']
   2. 加上Init_seen, 跑num_experiments次runExperiment()
   
4. **runExperiment():**
   
   1. 设置种子，以每次实验得到相同结果
   
   2. data.py / ```fetch_dataset()```:
      1. 通过eval初始化不同的datasets class =》datasets / movielens.py / class ML100K
         1. Movielens.py /```process()``` / make_explicit_data(), make_implicit_data(), make_info()处理数据
   
            1. Movielens.py /```make_explicit_data()```
               
               1. 100K的数据中column为user_id，movie_id，rating，unix_timestamp
               
               2. np.unique(): https://numpy.org/doc/stable/reference/generated/numpy.unique.html，https://blog.csdn.net/yangyuwen_yang/article/details/79193770
                  1. The sorted unique values.
                  2. The indices to reconstruct the original array from the unique array. Only provided if *return_inverse* is True. （旧列表元素在新列表中的位置）
               
               3. Csr_matrix:
               
                  1. csr_matrix((data, (row_ind, col_ind)), [shape=(M, N)]) (M*N的矩阵，每个格子都有一个值data[k]，只不过边的定位为(row_ind[k], col_ind[k]) )
               
                     where `data`, `row_ind` and `col_ind` satisfy the relationship `a[row_ind[k], col_ind[k]] = data[k]`.
               
            2. ```make_implicit_data()```
         
               1. 同make_explicit_data()
         
            3. ```make_info()```
   
               1. 将区间划分为数字
               1. 借用np.eye生成向量代表数字
               1. ```preprocessing.LabelEncoder().fit_transform()```, 将数据分类，获取代表的值(数字自增)
         
         2. ```.tocoo(), csr_matrix()```: 转稀疏矩阵为稠密矩阵, csc和csr有点像多叉树的序列化, 记录孩子的个数
         
            1. [coo_matrix](https://blog.csdn.net/The_Time_Runner/article/details/93636589) ：COOrdinate format matrix
            2. [csc_matrix](https://blog.csdn.net/The_Time_Runner/article/details/93640999) ：Compressed Sparse Column matrix, 记录每列第一个元素在values中出现的位置
            3. [csr_matrix](https://blog.csdn.net/The_Time_Runner/article/details/93641286) ：Compressed Sparse Row matrix，记录每行第一个元素在values中出现的位置
            4. Reference: https://www.cnblogs.com/zhangchaoyang/articles/5483453.html
         
      2. data.py /```make_pair_transform(dataset)```：datasets / utils.py / ```Compose(object)``` => data.py / Class PairInput(torch) :
      
         1. 前置知识: 实例化后，将实例当做函数调用会到```__call__()```, 例如a = A(), a() (调用```__call__()```)
         2. 前置知识: 如果class继承pytorch，然后a(params), 会调用```__call__()```, 而后```__call()___```中调用```forward()```, 并把参数传过去
         3. data.py / Class PairInput(torch) / ```forward(input)```: 
         4. 流程: dataset['train'].transform(input) => 在datasets / utils.py / Compose实例的```__call__()```中遍历 => 遍历的PairInput实例的```__call()__``` => PairInput实例的```forward()```
         4. dataset['train'].transform 会在遍历DataLoader =>```dataset, __getitem()__```时候调用
      
      3. data.py /```make_flat_transform(dataset)```：data.py / Class FlatInput(torch) => datasets / utils.py / ```Compose(object)```:
      
         1. 基本同```data.py / make_pair_transform(dataset)```
      
      4. dataset为dict, 其中dataset['train']和dataset['test']都为movielens.py的某一个class实例
      
   3. utils.py / ```process_dataset(dataset)```:
   
      1. hasattr() 函数用于判断对象是否包含对应的属性
      2. 增加cfg['data_size'], cfg['num_users']
      3. 增加cfg['info_size']
      4. 总的来说，给cfg增加一些key, value
   
   4. data.py / ```make_data_loader(dataset，cfg['model_name'](如ae))```:
   
      1. 遍历dataset的key, 此时key为'train'和'test', https://blog.csdn.net/loveliuzz/article/details/108756253
   
      2. torch.utils.data.DataLoader: https://www.cnblogs.com/dan-baishucaizi/p/14529897.html， https://pytorch-cn.readthedocs.io/zh/latest/package_references/data/
         1. 当在GPU上训练，pin_memory为True
   
      3. Collate_fn: https://blog.csdn.net/dong_liuqi/article/details/114521240, 
   
         collate_fn函数是实例化dataloader的时候, 以函数形式传递给loader.
   
         既然是collate_fn是以函数作为参数进行传递, 那么其一定有默认参数. 这个默认参数就是getitem函数返回的数据项的batch形成的列表. dataloader的输出是collate_fn的返回值
   
   5. models / ae.py
   
      1. 'models.ae().to(cfg["device"])' 通过eval初始化不同的model class =》datasets / ae.py / class ae
         1. 提取一些值,  ![image-20211204233610478](/Users/qile/Library/Application Support/typora-user-images/image-20211204233610478.png)，user_profile和item_attr是来自movielens.py / ```make_info()```的数据的column数，并且将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU或CPU上去，之后的运算都在GPU或CPU上进行。
         2. 初始化models / ae.py / class AE
            1. 初始化Encoder, Decoder。注意cfg['data_mode']为'user'时，初始化item, 反之亦然。
               1. 一些前置内容：http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html
               2. Pytorch基础知识在pytorch_explanation.md
               3. ae.py / class Encoder:
                  1. ```Torch.nn.Linear()```，https://pytorch.org/docs/stable/generated/torch.nn.Linear.html :
                     1. 设置全连接层
                  2. ```nn.Tanh()```: 双曲正切函数，一种激活函数，sigmoid也是一种激活函数
                  3. ```nn.Sequential(*blocks)```：
                     1. 2种激活方式: 按顺序放进去，或者用个```OrderedDict()```, https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
                     2. ```*``` is iterable unpacking notation in Python，https://stackoverflow.com/questions/66780615/what-does-the-sign-mean-in-this-nn-built-by-pytorch
                  4. ```nn.init.xavier_uniform_()``` : 是一个服从均匀分布的Glorot初始化器
                     1. 通过网络层时，输入和输出的方差相同，包括前向传播和后向传播
                     2. https://blog.csdn.net/luoxuexiong/article/details/95772045
                  5. ```Forward()```: 
               4. ae.py / class Decoder:
                  1. 基本同class Encoder
               5. 











Train:





Test:





Technology Issue:

exec: 内置语句，执行储存在字符串或文件中的Python语句，不返回结果

Vars: 内置函数，返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值 类似 locals()。

Eval: 内置函数，用于执行一个字符串表达式，并返回表达式的计算结果











1. cfg['num_organizations']
2. Ar, aw是什么  => assisted 不用考虑
3. Make_pair_transform() / make_flat_transform()
4. .to(CPU)





Data loading:













HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients:

