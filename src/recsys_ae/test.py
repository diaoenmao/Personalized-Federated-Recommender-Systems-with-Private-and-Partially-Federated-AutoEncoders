from config import cfg
from data import fetch_dataset, make_data_loader
from utils import collate, process_dataset, save_img, process_control, resume, to_device
import torch
import models

# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_name = 'NFP'
#     batch_size = {'train': 10, 'test': 10}
#     dataset = fetch_dataset(data_name)
#     data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         print(input['data'].shape)
#         print(torch.unique(input['data']))
#         exit()

# import numpy as np
# from scipy.sparse import csr_matrix
# import torch
#
# if __name__ == "__main__":
#     csr = csr_matrix([[1, 2.5, 0], [0, 0, 3.5], [4.5, 0, 5]])
#     print('csr', csr)
#     coo = csr.tocoo()
#     print('coo', coo)
#     t = torch.sparse_coo_tensor([coo.row.tolist(), coo.col.tolist()],
#                                 coo.data)
#     print('t', t)
#     z = csr[0].todense()
#     print('z', z)

# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_name = 'ML100K'
#     batch_size = {'train': 10, 'test': 10}
#     dataset = fetch_dataset(data_name)
#     data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#     for i, input in enumerate(data_loader['train']):
#         a = input['data'][0].tocoo()
#         t1 = torch.sparse_coo_tensor(torch.tensor([a.row.tolist(), a.col.tolist()]), torch.tensor(a.data))
#         b = input['data'][1].tocoo()
#         t2 = torch.sparse_coo_tensor(torch.tensor([a.row.tolist(), a.col.tolist()]), torch.tensor(a.data))
#         c = torch.cat([t1, t2], dim=0)[0]
#         print(c)
#         exit()

# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_name = 'ML100K'
#     batch_size = {'train': 10, 'test': 10}
#     dataset = fetch_dataset(data_name)
#     data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         # print(input['user'].size(), input['item'].size(), input['rating'].size(),
#         #       input['user_profile'].size(), input['item_attr'].size(),
#         #       input['target_user'].size(), input['target_item'].size(), input['target_rating'].size(),
#         #       input['target_user_profile'].size(), input['target_item_attr'].size())
#         print(input['user'].size(), input['rating'].size(),
#               input['user_profile'].size(), input['item_attr'].size(),
#               input['target_user'].size(), input['target_rating'].size(),
#               input['target_user_profile'].size(), input['target_item_attr'].size())
#         exit()

# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_names = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']
#     batch_size = {'train': 10, 'test': 10}
#     for data_name in data_names:
#         print(data_name)
#         dataset = fetch_dataset(data_name)
#         data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#         for i, input in enumerate(data_loader['train']):
#             input = collate(input)
#             print(input['user'][0].dtype, input['item'][0].dtype, input['target'][0].dtype)
#             break
#         for i, input in enumerate(data_loader['test']):
#             input = collate(input)
#             print(input['user'][0].dtype, input['item'][0].dtype, input['target'][0].dtype)
#             break


# import numpy as np
# from scipy.sparse import csr_matrix
# import torch
#
# if __name__ == "__main__":
#     data = [[1, 2.5, 0], [0, 0, 3.5], [4.5, 0, 5]]
#     csr = csr_matrix(data)
#     print('csr', csr)
#     print('csr', csr[2])
#     print('coo', csr[2].tocoo().col)
#     print('coo', csr[2].tocoo().data)

# import numpy as np
# from scipy.sparse import csr_matrix
# import torch
#
# if __name__ == "__main__":
#     data = [[0, 1, 0], [0, 0, 0], [4.5, 0, 5]]
#     csr = csr_matrix(data)
#     print('csr', csr)
#     print(csr[:3].tocoo().row)
#     print(csr[:3].tocoo().col)


# if __name__ == "__main__":
#     x = torch.full((5, 5), float('nan'))
#     x[0, 2:4] = 1
#     x[1, 1:3] = 2
#     x[2, [0, 2]] = 3
#     x[3, [1, 2]] = 4
#     x[4, [2, 4]] = 5
#     x = x[~x.isnan()].reshape(x.size(0), -1)
#     print(x)

# import numpy as np
# from scipy.sparse import csr_matrix
# if __name__ == "__main__":
#     M = 10
#     N = 20
#     row = np.arange(10)
#     col = np.arange(10)
#     data = np.ones(10)
#     csr = csr_matrix((data, (row, col)), shape=(M, N))
#     print('a', csr)
#     csr.data = np.ones(10) * 2
#     print('b', csr)


# import numpy as np
# from scipy.sparse import csc_matrix
# if __name__ == "__main__":
#     M = 10
#     N = 20
#     row = np.arange(10)
#     col = np.arange(10)
#     data = np.ones(10)
#     csc = csc_matrix((data, (row, col)), shape=(M, N))
#     print('a', csc.todense())
#     print('a', csc[:, :3].todense())

# def MAP(output, target, topk):
#     topk = min(topk, target.size(-1))
#     idx = torch.sort(output, dim=-1, descending=True)[1]
#     topk_idx = idx[:, :topk]
#     topk_target = target[torch.arange(target.size(0)).view(-1, 1), topk_idx]
#     precision = torch.cumsum(topk_target, dim=-1) / torch.arange(1, topk + 1).float()
#     m = torch.sum(topk_target, dim=-1)
#     ap = (precision * topk_target).sum(dim=-1) / m
#     map = ap.mean()
#     return map
#
#
# if __name__ == "__main__":
#     target = torch.tensor([1, 0, 1, 0, 0, 1, 0, 0, 1, 1]).view(1, -1)
#     output = torch.arange(target.size(-1), 0, -1).float().view(1, -1)
#     topk = 10
#     map = MAP(output, target, topk)
#     print(map)


# def parse_implicit_rating(user, item, output, target):
#     user, user_idx = torch.unique(user, return_inverse=True)
#     item, item_idx = torch.unique(item, return_inverse=True)
#     num_user, num_item = len(user), len(item)
#     output_rating = torch.full((num_user, num_item), -float('inf'), device=output.device)
#     target_rating = torch.full((num_user, num_item), -float('inf'), device=target.device)
#     output_rating[user_idx, item_idx] = output
#     target_rating[user_idx, item_idx] = target
#     return output_rating, target_rating
#
# if __name__ == "__main__":
#     user = torch.arange(10)
#     item = torch.arange(10)
#     output = torch.ones(10)
#     target = torch.zeros(10)
#     output_rating, target_rating = parse_implicit_rating(user, item, output, target)
#     print(output_rating)
#     print(target_rating)


# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     data_name = 'ML100K'
#     batch_size = {'train': 10, 'test': 10}
#     dataset = fetch_dataset(data_name)
#     data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#     for i, input in enumerate(data_loader['train']):
#         input = collate(input)
#         # print(input['user'].size(), input['item'].size(), input['rating'].size(),
#         #       input['user_profile'].size(), input['item_attr'].size(),
#         #       input['target_user'].size(), input['target_item'].size(), input['target_rating'].size(),
#         #       input['target_user_profile'].size(), input['target_item_attr'].size())
#         # print(input['user'].size(), input['item'].size(), input['rating'].size(),
#         #       input['item_attr'].size(),
#         #       input['target_user'].size(), input['target_item'].size(), input['target_rating'].size(),
#         #       input['target_item_attr'].size())
#         # print(input['user'].size(), input['item'].size(), input['rating'].size(),
#         #       input['user_profile'].size(), input['item_attr'].size(),
#         #       input['target_user'].size(), input['target_item'].size(), input['target_rating'].size())
#         # print(input['user'].size(), input['item'].size(), input['rating'].size(),
#         #       input['target_user'].size(), input['target_item'].size(), input['target_rating'].size())
#         exit()

# if __name__ == "__main__":
#     import numpy as np
#     from scipy.sparse import csr_matrix
#
#     row = np.arange(5)
#     col = np.arange(5)
#     data = np.ones(5)
#     t_1 = csr_matrix((data, (row, col)), shape=(10, 10))
#     row = np.arange(10)
#     col = np.arange(10)
#     data = np.ones(10)
#     data[-1] = 0
#     t_2 = csr_matrix((data, (row, col)), shape=(10, 10))
#     t_3 = 0 + t_1 + t_2
#     t_4 = 0 + t_2 + t_2
#     t_3[-1, -1] = 0
#     print(t_3)
#     print(t_4)
#     exit()
#     data = t_3.data / t_4.data
#     row = np.arange(10)
#     col = np.arange(10)
#     t_5 = csr_matrix((data, (row, col)), shape=(10, 10))
#     print(t_5)
#     exit()

# if __name__ == "__main__":
#     import numpy as np
#     from scipy.sparse import csr_matrix
#
#     row = np.arange(5)
#     col = np.arange(5)
#     data = np.ones(5)
#     t_1 = csr_matrix((data, (row, col)), shape=(10, 10))
#     print(t_1)
#     data = -np.ones(5)
#     t_1.data = data
#     print(t_1)


# if __name__ == "__main__":
#     process_control()
#     cfg['seed'] = 0
#     batch_size = {'train': 10, 'test': 10}
#     data_names = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'NFP']
#     for data_name in data_names:
#         dataset = fetch_dataset(data_name)
#         data_loader = make_data_loader(dataset, cfg['model_name'], batch_size=batch_size)
#         for i, input in enumerate(data_loader['train']):
#             input = collate(input)
#             print(input['user'].size(), input['item'].size(), input['rating'].size(),
#                   input['target_user'].size(), input['target_item'].size(), input['target_rating'].size())
#             break


# if __name__ == "__main__":
#     import torch
#     import numpy as np
#     from scipy.sparse import coo_matrix
#
#     coo = coo_matrix(([3, 4, 5], ([0, 1, 1], [2, 0, 2])), shape=(2, 3))
#
#     values = coo.data
#     indices = np.vstack((coo.row, coo.col))
#
#     i = torch.LongTensor(indices)
#     v = torch.FloatTensor(values)
#     shape = coo.shape
#
#     torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()
#
#     import time
#     import numpy as np
#     from scipy.sparse import csr_matrix
#
#     user = np.arange(1000).astype(np.int64)
#     item = np.arange(1000).astype(np.int64)
#     rating = np.ones(1000, dtype=np.float32) * 0.1
#     target_rating = np.ones(1000, dtype=np.float32) * 0.2
#     data = csr_matrix((rating, (user, item)), shape=(1000, 1000))
#     target = csr_matrix((target_rating, (user, item)), shape=(1000, 1000))
#     data_coo = data.tocoo()
#     target_coo = target.tocoo()
#     s = time.time()
#     input = {'user': torch.tensor(data_coo.row), 'item': torch.tensor(data_coo.col),
#              'rating': torch.tensor(data_coo.data), 'target_user': torch.tensor(target_coo.row),
#              'target_items': torch.tensor(target_coo.col), 'target_rating': torch.tensor(target_coo.data)}
#     # input = to_device(input, cfg['device'])
#     print('base', time.time() - s)
#     indices = torch.tensor(np.vstack((user, item)), dtype=torch.long)
#     values = torch.tensor(rating)
#     target_values = torch.tensor(target_rating)
#     data = torch.sparse_coo_tensor(indices, values, torch.Size([1000, 1000]))
#     target = torch.sparse_coo_tensor(indices, target_values, torch.Size([1000, 1000]))
#     s = time.time()
#     input = {'data': data, 'target': target}
#     input = to_device(input, cfg['device'])
#     print('new', time.time() - s)
#     time.sleep(65535)


# if __name__ == "__main__":
#     import torch
#     import numpy as np
#     from scipy.sparse import coo_matrix
#
#     coo = coo_matrix(([3, 4, 5], ([1, 1, 0], [2, 0, 2])), shape=(2, 3))
#     print(coo)
#     print(coo.row)
#     print(coo.col)
#     exit()

# if __name__ == "__main__":
#     import numpy as np
#     from scipy.sparse import csc_matrix
#     row = np.array([0, 2, 2, 0, 1, 2])
#     col = np.array([0, 0, 1, 2, 2, 2])
#     data = np.array([1, 2, 3, 4, 5, 6])
#     a = csc_matrix((data, (row, col)), shape=(3, 3))
#     b = a.transpose()
#     print(a.toarray())
#     print(b.toarray())
#     print(a[:, 0].toarray())
