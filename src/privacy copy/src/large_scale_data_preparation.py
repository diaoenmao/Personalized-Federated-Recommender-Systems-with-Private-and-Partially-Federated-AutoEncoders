import os
from config import cfg
# from datasets import ML100K, ML1M, ML10M, ML20M
# from datasets.datasets_utils import download_url, extract_file
# from utils import makedir_exist_ok, check_exists
# 
# file_list = [ML100K.file, ML1M.file, ML10M.file, ML20M.file]

dataset_list = ['ML100K', 'ML1M', 'ML10M', 'ML20M', 'Douban']
from utils import process_control
from data import fetch_dataset

def main():
    process_control()
    for i in range(len(dataset_list)):
        data_name = dataset_list[i]
        # fetch_dataset(data_name)
        # print('cfg!!!', cfg)
        fetch_dataset(data_name)
        # file = file_list[i]

        # root = os.path.join('data', '{}'.format(data_name))
        # root = os.path.expanduser(root)
        # raw_folder = os.path.join(root, 'raw')
        # print(f'root: {root}')
        # print(f'raw_folder: {raw_folder}')
        # if not check_exists(raw_folder):
        #     makedir_exist_ok(raw_folder)
        #     for (url, md5) in file:
        #         filename = os.path.basename(url)
        #         download_url(url, raw_folder, filename, md5)
        #         extract_file(os.path.join(raw_folder, filename))


if __name__ == "__main__":
    main()