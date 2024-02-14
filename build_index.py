import argparse
import json
import numpy as np
import hnswlib
import torch
from tqdm import tqdm

def cnt_index(season_id, episode_id):
    path = f'{root_path}/{season_id}/{episode_id}.npz'
    data = np.load(path)
    return len(data)

parser = argparse.ArgumentParser(description='Index features for episodes.')
parser.add_argument('-features_path', type=str, default='features', help='Path to the features directory (default: features)')
parser.add_argument('-conf_diff', type=str, default='conf_diff.json', help='Path to the configuration file for adding new features (default: conf_diff.json)')
parser.add_argument('-pre_index', type=str, default=None, help='Path to the previous feature index file (default: None)')
parser.add_argument('-dim', type=int, default=512, help='Dimension of the features (default: 512)')
parser.add_argument('-ef_construction', type=int, default=512, help='Balances construction time and index precision (default: 512)')
parser.add_argument('-m', type=int, default=64, help='Maximum number of outgoing connections in the graph (default: 64)')
parser.add_argument('-output', type=str, default='index.bin', help='Path to the output feature index file (default: index.bin)')
args = parser.parse_args()

conf = json.load(open(args.conf_diff, 'r', encoding='utf-8'))
seasons = conf['seasons']
root_path = args.features_path
num_elements = 0

for season in seasons:
    season_id = season['id']
    episodes = season['episodes']
    for episode in episodes:
        episode_id = episode['id']
        num = cnt_index(season_id, episode_id)
        num_elements += num

print('Total Key Frame Number:', num_elements)

p = hnswlib.Index(space='ip', dim=args.dim)
if args.pre_index is not None:
    p.load_index(args.pre_index)
    now_current_count = p.get_current_count()
    p.resize_index(now_current_count + num_elements)
else:
    p.init_index(max_elements=num_elements, ef_construction=args.ef_construction, M=args.m)

with tqdm(total=num_elements) as pbar:
    for season in seasons:
        season_id = season['id']
        episodes = season['episodes']
        for episode in episodes:
            episode_id = episode['id']
            path = f'{root_path}/{season_id}/{episode_id}.npz'
            features = np.load(path)
            
            ids = features.files
            index_range_start = episode['index_range'][0]
            data = []
            for id in ids:
                data.append(features[id])
            data_norm = torch.nn.functional.normalize(torch.from_numpy(np.array(data)), p=2, dim=1)
            for i in range(len(ids)):
                ids[i] = int(ids[i]) + index_range_start
            p.add_items(data_norm.numpy(), ids)
            pbar.update(len(features))
            pbar.set_description(f'Added {season_id} - {episode_id}')

p.save_index(args.output)
