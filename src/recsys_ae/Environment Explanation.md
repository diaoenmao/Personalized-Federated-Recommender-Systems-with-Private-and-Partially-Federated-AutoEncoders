mac VSCODE terminal python version与conda建立的python version对应不上的解决方法：

​	https://blog.csdn.net/neve_give_up_dan/article/details/112912913



conda create --name recsys_ae_conda python=3.8

conda activate recsys_ae_conda

pip install pyyaml

pip install torchvision

pip install scipy

pip install tensorboard

pip install anytree

pip install tqdm

pip install wheel

pip install pandas

pip install -U scikit-learn







删除conda环境:

conda env remove --name recsys_ae_conda



查看所有conda环境:

conda info --env



查看安装的包:

conda list