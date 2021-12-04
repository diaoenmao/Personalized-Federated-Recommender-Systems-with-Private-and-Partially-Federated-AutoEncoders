mac VSCODE terminal python version与conda建立的python version对应不上的解决方法：

​	https://blog.csdn.net/neve_give_up_dan/article/details/112912913

Vscode 测试with command, 添加"args", 修改"program":

​	https://code.visualstudio.com/docs/python/debugging

​    "program": "${workspaceFolder}/train_recsys_joint.py",

​	"args": [

​                "--control_name", "ML100K_user_implicit_ae_random-2_constant-0.1_constant_1",

​                "--num_workers", "0",

​                "--init_seed", "0",

​                "--num_experiments", "1", 

​                "--log_interval", "0.25",

​                "--device", "cpu", 

​                "--world_size", "1", 

​                "--resume_mode", "0", 

​                "--verbose", "False",

​      ]



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



