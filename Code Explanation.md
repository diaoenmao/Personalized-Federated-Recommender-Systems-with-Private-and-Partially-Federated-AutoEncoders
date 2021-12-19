数据集描述: https://grouplens.org/datasets/movielens/

1. 100K， https://files.grouplens.org/datasets/movielens/ml-100k-README.txt：
   1. 100,000 ratings (1-5) from 943 users on 1682 movies.
   2. user id | item id | rating | timestamp.





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
         
      2. data.py /```make_pair_transform(dataset)```：datasets / datasets_utils.py / ```Compose(object)``` => data.py / Class PairInput(torch) :
   
         1. 前置知识: 实例化后，将实例当做函数调用会到```__call__()```, 例如a = A(), a() (调用```__call__()```)
         2. 前置知识: 如果class继承pytorch，然后a(params), 会调用```__call__()```, 而后```__call()___```中调用```forward()```, 并把参数传过去
         3. data.py / Class PairInput(torch) / ```forward(input)```: 
         4. 流程: dataset['train'].transform(input) => 在datasets / datasets_utils.py / Compose实例的```__call__()```中遍历 => 遍历的PairInput实例的```__call()__``` => PairInput实例的```forward()```
         4. dataset['train'].transform 会在遍历DataLoader =>```dataset, __getitem()__```时候调用
   
      3. data.py /```make_flat_transform(dataset)```：data.py / Class FlatInput(torch) => datasets / datasets_utils.py / ```Compose(object)```:
   
         1. 基本同```data.py / make_pair_transform(dataset)```
   
      4. dataset为dict, 其中dataset['train']和dataset['test']都为movielens.py的某一个class实例
   
   3. utils.py / ```process_dataset(dataset)```:
   
      1. hasattr() 函数用于判断对象是否包含对应的属性
      2. 增加cfg['data_size'], cfg['num_users']
      3. 增加cfg['info_size']
      4. 总的来说，给cfg增加一些key, value
   
   4. data.py / ```make_data_loader(dataset，cfg['model_name'](如ae))```:
   
      1. 遍历dataset的key, 此时key为'train'和'test', https://blog.csdn.net/loveliuzz/article/details/108756253 (总DataLoader流程)
   
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
                     1. A sequential container. Modules will be added to it in the order they are passed in the constructor. Alternatively, an ordered dict of modules can also be passed in.
                        一个有序的容器，神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行，同时以神经网络模块为元素的有序字典也可以作为传入参数。
                     2. 2种激活方式: 按顺序放进去，或者用个```OrderedDict()```, https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
                     3. ```*``` is iterable unpacking notation in Python，https://stackoverflow.com/questions/66780615/what-does-the-sign-mean-in-this-nn-built-by-pytorch
                  4. ```nn.init.xavier_uniform_()``` : 是一个服从均匀分布的Glorot初始化器
                     1. 通过网络层时，输入和输出的方差相同，包括前向传播和后向传播
                     2. https://blog.csdn.net/luoxuexiong/article/details/95772045
                  5. ```Forward(input)```: 
                     1. 传入input到模型， 得到结果
               4. ae.py / class Decoder:
                  1. 基本同class Encoder
               5. 设置nn.Dropout, https://www.jb51.net/article/212770.htm
               6. 如果设置了info_size，说明要考虑side information，初始化side information的Encoder
            2. ```forward(input)```
               1. torch.no_grad(): Context-manager that disabled gradient calculation. 
                  1. 意思是被这个包住的代码进行计算的时候不会记录梯度，省略计算资源
                  2. https://pytorch.org/docs/stable/generated/torch.no_grad.html
                  3. https://zhuanlan.zhihu.com/p/386454263
                  4. 处理Encoder的Input
               2. 将Input传入basic的encoder, 处理Encoder result, 如果要考虑side information, 将side information的encoder的结果加入basic的encoder的结果
               3. dropout结果
               4. 将encoder的结果传入decoder
               5. 处理结果
                  1. 取反？
                  2. 将loss加入output, loss在models / models_utils.py / loss_fn(output, target, reduction) 中定义
                     1. 如果cfg['target_mode']为implicit, 使用cross_entropy
                     2. 如果cfg['target_mode']为explicit, 使用explicit
   
   6. Utils.py / ```make_optimizer(model, cfg['model_name'])``` :
   
      1. 优化器就是需要根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用。
      2. https://zhuanlan.zhihu.com/p/95976265
      3. 返回optimizer instance
   
   7. Utiles.py / ```make_scheduler(optimizer, cfg['model_name'])```:
   
      1. 设定训练时如何改变optimizer的参数的scheduler
      1. 返回scheduler instance
   
   8. metrics / class Metric:
   
      1. 处理衡量结果的类
      2. 核心点:
         1. 将初始信息，包括初始值，方向，名字赋予self.pivot, self.pivot_direction, self.pivot_name
         2. 将衡量函数的实例赋予到self.metric的key中
   
   9. Utils.py / ```resume()```:
   
      1. 处理恢复训练的情况
   
   10. 使用多个GPU加速训练: 
   
       1. https://zhuanlan.zhihu.com/p/102697821，当你调用nn.DataParallel的时候，只是在你的input数据是并行的，但是你的output loss却不是这样的，每次都会在第一块GPU相加计算，这就造成了第一块GPU的负载远远大于剩余其他的显卡
   
   11. 正式训练
   
       1. Train / test 模型```cfg[cfg['model_name']]['num_epochs']```这么多轮
       2. train_privacy_joint.py / Train(data_loader, model, optimizer, metric, logger, epoch)
          1. 将Model设置为taining mode
          2. 遍历batch, 具体data_loader流程: https://mp.weixin.qq.com/s/Uc2LYM6tIOY8KyxB7aQrOw, Transforms => Sampler => Collate Function. (应该是sampler => Transforms，这个顺序错了)
             1. Dataset是一个包装类，用来将数据包装为Dataset类，然后传入DataLoader中, DataLoader生成的实例为可迭代对象
             2. 完整流程: 
                1. 遍历DataLoader (调用```__iter__()```获取迭代器, ```__next()```遍历)
                2. 利用index去Sampler实例中取对应的index (```__next_index()```)
                3. 利用取到的index去Dataset实例中取对应的数据, Dataset实例中```__getitem()__```的调用实例的transform
                4. transform(input) => 在datasets / datasets_utils.py / Compose实例的```__call__()```中遍历 => 遍历的PairInput实例的```__call()__``` => PairInput实例的```forward()```
             3. 图示: https://blog.csdn.net/loveliuzz/article/details/108756253 (总DataLoader流程)
             4. 更新参数
                1. torch.nn.utils.clip_grad_norm_， 梯度截断: https://blog.csdn.net/weixin_42628991/article/details/114845018, https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
             5. 记录metric
             6. 记录详细信息当epoch是某个数的倍数





Technology Issue:

exec: 内置语句，执行储存在字符串或文件中的Python语句，不返回结果

Vars: 内置函数，返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值 类似 locals()。

Eval: 内置函数，用于执行一个字符串表达式，并返回表达式的计算结果



要做的事情:

1. 修改data_loader
2. 加入federated



小问题:

1. (done) cfg['num_organizations'] => split dataset 用
2.  (done) Ar, aw是什么  => assisted 不用考虑
3. Make_pair_transform() / make_flat_transform() => 在compose中对数据进一步处理
4. .to(CPU)
4. 为什么要ae.py / class AE / with torch.no_grad()，好像没什么用
4. loss对Implicit, explicit不同的原因

7. cfg['world_size'] 是什么
8. movielens.py / make_explicit_data() => 有几行代码没什么用





没看的部分:

1. 处理side information没看
2. metrics / class Metric中存的RMSE, Accuracy, MAP类没看
3. utils.py / resume没看
4. Logger.py / make_logger 没看









------------------------





HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients:

