import argparse
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, split_dataset, make_split_dataset
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger

parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    cfg['data_mode'] = 'user'
    cfg['info'] = 1
    cfg['data_split_mode'] = 'genre'
    cfg['num_organizations'] = 18
    data_names = ['ML100K', 'ML1M', 'ML10M']
    for data_name in data_names:
        print(data_name)
        cfg['data_name'] = data_name
        dataset = fetch_dataset(cfg['data_name'])
        print('rating', dataset['train'].data.shape)
        if hasattr(dataset['train'], 'user_profile'):
            print('user profile', dataset['train'].user_profile['data'].shape)
        if hasattr(dataset['train'], 'item_attr'):
            print('item attribute', dataset['train'].item_attr['data'].shape)
        process_dataset(dataset)
        data_split = split_dataset(dataset)
        data_split_count = [len(x) for x in data_split]
        print('Genre', data_split_count)
    genre_list = ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                  'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War',
                  'Western']
    print(genre_list)
    return

if __name__ == "__main__":
    main()
