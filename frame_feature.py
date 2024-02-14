import cv2
import torch
import numpy as np
from torchvision import transforms
from model import Search_Model
import imagehash
from PIL import Image
import json
import os
import argparse
from tqdm import tqdm
import concurrent.futures

parser = argparse.ArgumentParser(description="Extract frame features using a pre-trained model.")
parser.add_argument("-model_path", default="model.pth", help="Path to the pre-trained model file (default: model.pth)")
parser.add_argument("-conf_path", default="conf_diff.json", help="Path to the configuration file (default: conf.json)")
parser.add_argument("-info_path", default="info.json", help="Path to the info file (default: info.json)")
parser.add_argument("-output", default="features", help="Output path for extracted features (default: features)")
parser.add_argument("-threads", type=int, default=4, help="Number of threads for parallel processing (default: 4)")
args = parser.parse_args()
thread_num = args.threads
conf = json.load(open(args.conf_path, 'r', encoding="utf-8"))
info = json.load(open(args.info_path, 'r', encoding="utf-8"))
settings = conf['settings']
fps = settings.get('fps', 8)
op_length = settings.get('op_length', 0)
ed_length = settings.get('ed_length', 0)
full_first_ep = settings.get('full_first_ep', True)
sim_threshold = settings.get('sim_threshold', 4)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Search_Model().to(device)
if os.path.exists(args.model_path):
    model.load_state_dict(torch.load(args.model_path))
    print(f"Loaded model from {args.model_path}")
model.eval()

output_path = args.output
if not os.path.exists(output_path):
    os.makedirs(output_path)
def check_sim(phash_list, phash, max_size = 100):
    for p in phash_list:
        if p - phash <= sim_threshold:
            return True
    phash_list.insert(0, phash)
    if len(phash_list) > max_size:
        phash_list.pop()
    return False
def process_video(video_path, fps, season_id, episode_id):
    save_path = os.path.join(output_path, f"{season_id}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_path = os.path.join(save_path, f"{episode_id}.npz")
    if os.path.exists(save_path):
        print(f"{save_path} already exists, skipping...")
        return
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=info['mean'], std=info['std']),
    ])

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    frame_list = []
    while True:
        target = int(i * (1 / fps) * video_fps)
        i += 1
        if target > total_frames:
            break
        frame_list.append(target)

    # 用于存储推理结果的字典
    results_dict = {}

    phash_list = []
    if not full_first_ep or episode_id > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(op_length * video_fps))
        frame_count = int(op_length * video_fps)
        end_frame = int(total_frames - ed_length * video_fps)
    else:
        frame_count = 0
        end_frame = total_frames

    with tqdm(total=end_frame - frame_count, desc=f"Processing {season_id}-{episode_id} Frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            pbar.update(1)
            if frame_count >= end_frame:
                break
            if frame_count in frame_list:
                # 图像预处理和推理
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                phash = imagehash.phash(img)
                
                if check_sim(phash_list, phash):
                    frame_count += 1
                    continue
                img = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img)
                frame_index = frame_list.index(frame_count)
                # 存储推理结果
                results_dict[str(frame_index)] = output.cpu().numpy()
            frame_count += 1
    # 释放视频对象
    cap.release()

    # 将推理结果保存到npz文件
    np.savez_compressed(save_path, **results_dict)

def process_video_wrapper(args):
    process_video(*args)

if __name__ == "__main__":
    seasons = conf['seasons']
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_num) as executor:
        tasks = [(episode['path'], fps, season['id'], episode['id']) for season in seasons for episode in season['episodes']]
        executor.map(process_video_wrapper, tasks)