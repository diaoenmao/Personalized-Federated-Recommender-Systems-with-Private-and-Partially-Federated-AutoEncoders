import pandas as pd
import numpy as np
import os

data_i = pd.read_csv('combined_data_2.txt', delimiter=',', names=["movie_id", "rating", "timestamp", 'user_id'])

total_item = data_i.shape[0]
print('total_item', total_item)
# print(f'data_i: {data_i}')
# print(f'dir: {dir(data_i)}')

index = data_i[data_i['rating'].isnull()].index
print(f'index: {index}')

user_id = data_i.iloc[index, 0].to_numpy()
print(f'user_id: {user_id}', len(user_id))


# b = user_id.view((str,1)).reshape(len(user_id),-1)[:,:,-1]
# user_id = np.fromstring(b.tostring(),dtype=(str,end-start))

for i in range(len(user_id)):
    user_id[i] = user_id[i][:-1]
print(f'user_id: {user_id}')
numpy_index = list(index)
# numpy_index.pop(0)
numpy_index.append(total_item)
print('numpy_index:', len(numpy_index))

extend_user_id = []
for i in range(len(user_id)):
    cur_user_id = user_id[i]
    cur_user_id_extend = [cur_user_id] * (numpy_index[i+1] - numpy_index[i] - 1)
    extend_user_id.extend(cur_user_id_extend)

print('extend_user_id:', len(extend_user_id))
extend_user_id = np.array(extend_user_id)
print(extend_user_id)

data_i = data_i.drop(index)
# print(f'after: data_i: {data_i}')

movie_id = data_i.iloc[:, 0].to_numpy()
rating = data_i.iloc[:, 1].to_numpy()
print('movie_id:', len(movie_id))

# data_i = data_i.drop(index)
# print(f'after: data_i: {data_i}')
# print(data_i['movie_id'][index])

# import copy
# data_i['user_id'][index] = copy.deepcopy(data_i['movie_id'][index])
# print(f'data_i: {data_i}')