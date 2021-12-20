
from scipy import sparse
import numpy as np
import torch
a = torch.tensor(np.array([0]), dtype=torch.long)
b = torch.tensor(np.array([1]), dtype=torch.long)
c = []
c.append(a)
c.append(b)

d = torch.cat(c, 0)
print(d, type(d))
# a = sparse.coo_matrix((3, 4), dtype=np.int8).toarray()

ceshi = torch.full((10, 10), float('10'))
rating = torch.full((10, 10), float('nan'))
rating[0][1] = 5

rating_mask = ~(rating.isnan())
print(rating_mask)
print(ceshi[rating_mask])

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
