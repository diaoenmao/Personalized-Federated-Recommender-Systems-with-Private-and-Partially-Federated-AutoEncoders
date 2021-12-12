import numpy as np
from collections.abc import Iterator, Iterable
# a = [0,1,2]
# b = np.array([4,2,3,4,5])[a]
# print(a)
# print(b)

# user = np.array([1, 2, 2, 5, 3, 4, 3])
# item = np.array([1])
# # user_id: all sorted unique user_id
# # user_inv: The index of (old array[i] in new array)
# user_id, user_inv = np.unique(user, return_inverse=True)
# print("1", user_id, user_inv)
# item_id, item_inv = np.unique(item, return_inverse=True)
# M, N = len(user_id), len(item_id)

# # key: unique user id
# # val: index
# user_id_map = {user_id[i]: i for i in range(len(user_id))}
# print("2", user_id_map)
# item_id_map = {item_id[i]: i for i in range(len(item_id))}

# # np.array(): index array
# # np.array()[user_inv]: 还是user_inv?
# a = np.array([user_id_map[i] for i in user_id], dtype=np.int64)
# print("3", a)
# user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
# print("4", user)
# item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)


# print('models.{}().to(cfg["device"])'.format('zzzzz'))

# def func():
#     yield 1

# obj = func()
# print(obj.__iter__())
# print("----", obj.__iter__() == obj)
# print(next(obj))
# print(isinstance(obj, Iterator))
# print(isinstance(obj, Iterable))

# a = [1,2,3]
# print(a.__iter__())
# print("----", a.__iter__() == a)
# b = a.__iter__()
# print(next(b))


# f = open('./ceshi.txt','r') # opening a file
# print(f.readlines())

import logging
import os
import errno

def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return  

def generate_logger(log_path):
    logger = logging.getLogger('Apollo_logger')

    if not logger.handlers:
        logger.setLevel(level=logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        handler = logging.FileHandler(log_path)
        handler.setFormatter(formatter)
        
        logger.addHandler(handler)

        # 输出到窗口
        # handler = logging.StreamHandler(sys.stdout)
        # handler.setFormatter(formatter)
        # logger.addHandler(handler)
    
    return logger

def log(msg, self_id, task_id, test_id=None):
    
    print("~~~~",msg)
    root = os.path.abspath(os.path.dirname(__file__))
    root = os.path.join(root, 'log_file')

    self_id = str(self_id)
    task_id = str(task_id)
    if test_id:
        test_id = str(test_id)

    log_path = None 
    if test_id is None:
        makedir_exist_ok(os.path.join(root, self_id, 'task', task_id, 'train'))
        log_path = os.path.join(root, self_id, 'task', task_id, 'train', 'current_task.log')
    else:
        makedir_exist_ok(os.path.join(root, self_id, 'task', task_id, 'test', test_id))
        log_path = os.path.join(root, self_id, 'task', task_id, 'test', test_id, 'current_test.log')
    
    print("log_path-------------------------", log_path, type(log_path))
    logger = generate_logger(log_path)
    logger.debug(msg)

    return

import sys
# from logging.handlers import TimedRotatingFileHandler
def get_log(self_id, task_id, test_id=None):

    """
    read txt file and return content of txt file.

    Parameters:
       self_id - id of current user
       task_id - task_id of task
       test_id - test_id of test

    Returns:
        data - List. ['first log_interval\n', 'second\n', 'third']

    Raises:
        KeyError - raises an exception
    """    

    root = os.path.abspath(os.path.dirname(__file__))
    root = os.path.join(root, 'log_file')

    self_id = str(self_id)
    task_id = str(task_id)
    if test_id:
        test_id = str(test_id)

    log_path = None
    if test_id is None:
        log_path = os.path.join(root, self_id, 'task', task_id, 'train', 'current_task.log')
        f = open(log_path, "r")
        return f.readlines()
            
    else:
        log_path = os.path.join(root, self_id, 'task', task_id, 'test', test_id, 'current_test.log')
        f = open(log_path, "r")
        return f.readlines() 



log("zzz", 1, 1)
log("ddd", 1, 1)
log("aaa", 1, 1)
log("bbb", 1, 1)

print(get_log(1,1))