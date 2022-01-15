
from scipy import sparse
import numpy as np
from scipy.sparse import csr_matrix
# import torch
# a = torch.tensor(np.array([0]), dtype=torch.long)
# b = torch.tensor(np.array([1]), dtype=torch.long)
# c = []
# c.append(a)
# c.append(b)

# d = torch.cat(c, 0)
# print(d, type(d))
# # a = sparse.coo_matrix((3, 4), dtype=np.int8).toarray()

# ceshi = torch.full((10, 10), float('10'))
# rating = torch.full((10, 10), float('nan'))
# rating[0][1] = 5

# rating_mask = ~(rating.isnan())
# print(rating_mask)
# print(ceshi[rating_mask])

# row = np.array([0, 3, 1, 0])
# col = np.array([0, 3, 1, 2])
# data = np.array([6, 5, 7, 8])
# b = sparse.csr_matrix((data, (row, col)), shape=(4, 4))
# print(b)
# # print(dir(b))
# print(b.indptr)
# print(b.indices)
# print(b[0])
# b.tocoo()
# # print(b)
# user = np.array([6, 7, 6,7,1,2])
# user_id, user_inv = np.unique(user, return_inverse=True)
# print("zz", user_id, user_inv)
# # item_id, item_inv = np.unique(item, return_inverse=True)
# # M, N = len(user_id), len(item_id)
# M = len(user_id)

# # # key: unique user id
# # # val: index
# user_id_map = {user_id[i]: i for i in range(len(user_id))}
# print("zz2", user_id_map)
# # item_id_map = {item_id[i]: i for i in range(len(item_id))}

# # np.array(): index array
# # np.array()[user_inv]: 还是user_inv?
# user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
# print("zz3", user)
# # item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)

# a = np.array([1, 2, 6, 4, 2, 3, 2])
# u, indices = np.unique(a, return_inverse=True)
# print(u, indices)

# class A:
#     def __init__(self, val=None):
#         b = val
#         pass

# # def generate(i):
# #     b = A(i)
# #     print(id(b))

# # for i in range(5):
# #     a = A()
# import pickle
# a = A()
# b = A()
# c = A()
# d = A()
# e = A()

# print(id(a))
# print(id(b))
# print(id(c))
# print(id(d))
# print(id(e))

# x = 5
# print(id(x))
# x -= 1
# print(id(x))
# x = x - 1
# print(id(x))




# test_user, test_item, test_rating = np.array([1,3,4]), np.array([2,5,8]), np.array([6,7,8])
# test_target = csr_matrix((test_rating, (test_user, test_item)), shape=(5,9))
# print(test_target)
# print("!!", test_target[0])
# print("!!", test_target[1])
# print("!!", test_target[2])
# print("!!", test_target[3])
# print("!!", test_target[4], test_target.shape[0])
# import torch
# from torch.utils.data import DataLoader, Dataset

# a = torch.tensor(np.array([1,2,3,4,5,6,7,8]))

# b = DataLoader(dataset=a, batch_size=2, shuffle=True)
# for i, v in b:
#   print(i)
#   print(v)
  # print(k)
# pickle.dump(input, open(path, 'wb'))


# def update(self, client):
#     with torch.no_grad():
#         valid_client = [client[i] for i in range(len(client)) if client[i].active]
#         if len(valid_client) > 0:
#             model = eval('models.{}()'.format(cfg['model_name']))
#             model.load_state_dict(self.model_state_dict)
#             global_optimizer = make_optimizer(model, 'global')
#             global_optimizer.load_state_dict(self.global_optimizer_state_dict)
#             global_optimizer.zero_grad()
#             weight = torch.ones(len(valid_client))
#             weight = weight / weight.sum()
#             for k, v in model.named_parameters():
#                 parameter_type = k.split('.')[-1]
#                 if 'weight' in parameter_type or 'bias' in parameter_type:
#                     tmp_v = v.data.new_zeros(v.size())
#                     for m in range(len(valid_client)):
#                         tmp_v += weight[m] * valid_client[m].model_state_dict[k]
#                     v.grad = (v.data - tmp_v).detach()
#             global_optimizer.step()
#             self.global_optimizer_state_dict = global_optimizer.state_dict()
#             self.model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
#         for i in range(len(client)):
#             client[i].active = False
import json
# a = [[1],[2]]
# b = json.dumps(a)
# c = '{b}'.format(b = b)

# print(c, type(c))
# d = json.loads(c)
# print(d, type(d))


# import hashlib
# input = "xiexianjian"
# output = hashlib.sha256(str(input).encode('utf-8'))
# output = output.hexdigest()
# print(output)
import torch
v = torch.arange(5)
b = v.new_zeros(v.size(), dtype=torch.float32)

print(v, b, id(v), id(b))