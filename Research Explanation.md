1. 资料: https://developers.google.com/machine-learning/recommendation

   

2. https://github.com/hongleizhang/RSPapers

   https://arxiv.org/pdf/2010.01264.pdf

   

1. Recommandation System

   1. Auto-encoder + federative learning 
   2. optimize auto-encoder 结构
   3. 基于auto-encoder进行分布式学习

   基于 auto-encoder: 分析评分。 用户评分完毕，根据评分历史预测没有打分的电影。



Federated Learning 算法 (简化版):

1. enable the training of heterogeneous local models with varying computation com- plexities and still produce a single global inference model.

   

Assisted 算法 (简化版):

1. 假设有m个domain，先初始化
2. 每一个domain计算自身的residual, 而后传输与each of the m-1 domain交集items的residual, respectively
3. 每一个domain aggregates (2中计算自身的residual) and (residuals from other domains), 形成一个所有residuals的target list
4. 每一个domain寻找最小的losslocal AAE，target为上面的target list



问题:

1. nmf, gmf 是什么 (nmf是ncf, gmf没效果)
2. objective (loss function), metric (accuracy)
2. fmin在更新什么？怎么更新的？(backpropagation更新模型)



新research不同:

1. 不需要互相传播residual   =》 删代码
2. 训练based on自己的result  =》 改target
3. column数一样，不需要调整 =》 改代码
4. 多加一个对decoder average =》加函数



Difference:

Federated: sharding, horizontal split

Assisted: vertical split







