import numpy as np
# a = [0,1,2]
# b = np.array([4,2,3,4,5])[a]
# print(a)
# print(b)

user = np.array([1, 2, 2, 5, 3, 4, 3])
item = np.array([1])
# user_id: all sorted unique user_id
# user_inv: The index of (old array[i] in new array)
user_id, user_inv = np.unique(user, return_inverse=True)
print("1", user_id, user_inv)
item_id, item_inv = np.unique(item, return_inverse=True)
M, N = len(user_id), len(item_id)

# key: unique user id
# val: index
user_id_map = {user_id[i]: i for i in range(len(user_id))}
print("2", user_id_map)
item_id_map = {item_id[i]: i for i in range(len(item_id))}

# np.array(): index array
# np.array()[user_inv]: 还是user_inv?
a = np.array([user_id_map[i] for i in user_id], dtype=np.int64)
print("3", a)
user = np.array([user_id_map[i] for i in user_id], dtype=np.int64)[user_inv].reshape(user.shape)
print("4", user)
item = np.array([item_id_map[i] for i in item_id], dtype=np.int64)[item_inv].reshape(item.shape)