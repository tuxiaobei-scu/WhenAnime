import gradio as gr

import json
from bisect import bisect_left
import hnswlib
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch
from model import Search_Model
import cv2
import imageio
import time
from urllib.parse import quote

video_path_root = os.environ.get('video_path_root', '')
ef = int(os.environ.get('ef', 512))
index_path = os.environ.get('index_path', 'index.bin')
conf_path = os.environ.get('conf_path', 'conf.json')
info_path = os.environ.get('info_path', 'info.json')
model_path = os.environ.get('model_path', 'model.pth')

info = json.load(open(info_path, 'r', encoding="utf-8"))
conf = json.load(open(conf_path, 'r', encoding='utf-8'))
fps = conf['settings']['fps']
seasons = conf['seasons']
index_id_list = []
episodes_list = []



video_path = {}
video_fps = {}
season_index_range = {}
for season in seasons:
    season_id = season['id']
    season_name = season['name']
    episodes = season['episodes']
    video_path[season_id] = {}
    video_fps[season_id] = {}
    season_index_range[season_id] = season['index_range']
    for episode in episodes:
        episode['season_id'] = season_id
        episode['season_name'] = season_name
        video_path[season_id][episode['id']] = episode['path']
        video_fps[season_id][episode['id']] = episode['fps']
        episodes_list.append(episode)
        index_id_list.append(episode['index_range'][1])

p = hnswlib.Index(space = 'ip', dim = 512) # possible options are l2, cosine or ip
p.load_index(index_path)
p.set_ef(ef)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=info['mean'], std=info['std']),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Search_Model().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
def query_img(img, query_num = 10, season_filter_data=[]):
    img = img.convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        query = model(img)
    query = torch.nn.functional.normalize(query, p=2, dim=0)
    query = query.cpu().numpy()
    if len(season_filter_data) == 0:
        labels, distances = p.knn_query(query, k = query_num)
    else:
        def filter(idx):
            for index_range in season_filter_data:
                if idx >= index_range[0] and idx <= index_range[1]:
                    return True
            return False
        labels, distances = p.knn_query(query, k = query_num, filter=filter, num_threads=1)
    return labels[0], distances[0]

def get_result(labels, distances):
    result = []
    for i in range(len(labels)):
        ep_index = bisect_left(index_id_list, labels[i])
        ep_info = episodes_list[ep_index]
        tim = (labels[i] - ep_info['index_range'][0]) / fps
        result.append({
            'season_id': ep_info['season_id'],
            'season_name': ep_info['season_name'],
            'episode_id': ep_info['id'],
            'episode_name': ep_info['name'],
            'time': tim,
            'similarity': (1 - distances[i]) * 100
        })
    return result


# 定义查询函数
def query_and_get_result(img, query_num=10, season_filter_data=[]):
    if img is None:
        gr.Warning("请上传图片")
        return []
    img = Image.fromarray(img)
    labels, distances = query_img(img, int(query_num), season_filter_data)
    result = get_result(labels, distances)
    return result

def get_season_filter_data(season_filter_str):
    if season_filter_str == '':
        return []
    res = []
    season_filter = season_filter_str.split(',')
    for i in range(len(season_filter)):
        if season_filter[i].isdigit():
            sid = int(season_filter[i])
            if sid in season_index_range:
                res.append(season_index_range[sid])
            else:
                gr.Warning("搜索范围不合法(将忽略范围搜索)")
                return []
        else:
            p = season_filter[i].split('-')
            if len(p) == 2:
                if p[0].isdigit() and int(p[0]) in season_index_range and p[1].isdigit() and int(p[1]) in season_index_range:
                    res.append([season_index_range[int(p[0])][0], season_index_range[int(p[1])][1]])
                else:
                    gr.Warning("搜索范围不合法(将忽略范围搜索)")
                    return []
            else:
                gr.Warning("搜索范围不合法(将忽略范围搜索)")
                return []
    return res

def cleanup_tmp_directory(directory, max_age_hours=2):
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            creation_time = os.path.getctime(filepath)
            if current_time - creation_time > max_age_seconds:
                os.remove(filepath)
                print(f"Deleted file: {filepath}")
def get_img(season, episode, raw_time):
    if video_path_root.startswith('http'):
        video_path_root + quote(video_path[int(season)][int(episode)])
    else:
        video_file = video_path_root + video_path[int(season)][int(episode)]
    cap = cv2.VideoCapture(video_file)
    vfps = video_fps[int(season)][int(episode)]
    start_frame = int((float(raw_time) - 0.5) * vfps)
    end_frame = int((float(raw_time) + 1.5) * vfps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_path = os.path.join(os.path.join(os.path.dirname(__file__), 'tmp'))
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    cleanup_tmp_directory(output_path, 2)
    output_file = os.path.join(output_path, f'/{season}_{episode}_{end_frame}.gif')
    frames = []

    ok = False

    # Read and write frames within the desired segment
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ok = True
            frame = cv2.resize(frame, (640, 360))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_frame >= end_frame:
                break
        else:
            break

    # Save frames as GIF
    imageio.mimsave(output_file, frames, fps=vfps, loop=0)

    # Release resources
    cap.release()

    if not ok:
        raise gr.Error("获取预览视频失败")
    return output_file

max_textboxes = 50



with gr.Blocks() as demo:
    header_html = """
        <h1>以图搜番</h1>
        <h3>By tuxiaobei</h3>
        <ol>
            <li>基于对比学习，训练适用于喜灰画风的检索模型，特征提取后用 HNSW 算法进行特征检索</li>
            <li>训练中已针对图片裁剪、添加滤镜、图片翻转的情况进行了训练，具有一定的鲁棒性。但过度裁剪、变换颜色仍可能造成搜索结果不准确，可尝试指定搜索范围提高准确性</li>
            <li>模型目前对弹幕、过多水印、黑边等干扰的处理能力有限，建议裁剪去除黑边等无关内容</li>
            <li>视频预览加载时间较长，有可能拉取视频失败，若失败请稍后重试</li>
        </ol>
    """
    header = gr.HTML(value=header_html)

    with gr.Row():
        with gr.Column(scale=1):
            img = gr.Image(height=360)
            """
            gr.Examples(
                examples=[os.path.join(os.path.dirname(__file__), "examples/1.jpg"), 
                          os.path.join(os.path.dirname(__file__), "examples/2.jpg"),
                          os.path.join(os.path.dirname(__file__), "examples/3.jpg"),
                          os.path.join(os.path.dirname(__file__), "examples/4.jpg"),
                          os.path.join(os.path.dirname(__file__), "examples/5.jpg")],
                inputs=img,
            )
            """
        with gr.Column(scale=1):
            num_input = gr.Slider(1, max_textboxes, value=10, step=1, label="搜索结果数量")
            season_filter = gr.Textbox(label="搜索范围（按季过滤）", info="可使用 , 和 - 分隔，如：19,20,22-24，表示第 19,20,22,23,24 季。若为空则表示不限制。")
            with gr.Row():
                html1 = """
                <ul>
                <li>第 19 部：羊村守护者</li>
                <li>第 20 部：跨时空救兵</li>
                <li>第 21 部：奇趣外星客</li>
                <li>第 22 部：异国大营救</li>
                <li>第 23 部：筐出胜利</li>
                </ul>
                """
                html2 = """
                <ul>
                <li>第 24 部：决战次时代</li>
                <li>第 25 部：奇妙大营救</li>
                <li>第 26 部：勇闯四季城</li>
                <li>第 27 部：遨游神秘洋</li>
                <li>第 28 部：心世界奇遇</li>
                </ul>
                """
                # season_filter_html1 = gr.HTML(html1)
                # season_filter_html2 = gr.HTML(html2)
            submit_btn = gr.Button("搜索，启动！")
        
            

    with gr.Column() as output_col:
        rows = []
        for i in range(max_textboxes):
            with gr.Row(visible=(i == 0)) as row:
                row_data = {}
                with gr.Column(scale=2):
                    row_data['season'] = gr.Textbox(label="季")
                    row_data['episode'] = gr.Textbox(label="集")
                    row_data['time'] = gr.Textbox(label="时间")
                with gr.Column(scale=2):
                    row_data['rank'] = gr.Textbox(label="排名")
                    row_data['sim'] = gr.Textbox(label="相似度")
                    get_img_btn = gr.Button(f"获取结果 #{i + 1} 预览视频")
                with gr.Column(scale=3):
                    row_data['video'] = gr.Image(label=f"预览视频 #{i + 1}", type='filepath')
                row_data['season_id'] = gr.Textbox(visible=False)
                row_data['episode_id'] = gr.Textbox(visible=False)
                row_data['raw_time'] = gr.Textbox(visible=False)

                row_data['row'] = row
                get_img_btn.click(get_img, inputs=[row_data['season_id'], row_data['episode_id'], row_data['raw_time']], outputs=row_data['video'], concurrency_limit=8)
                rows.append(row_data)

    def submit(img, query_num, season_filter):
        season_filter_data = get_season_filter_data(season_filter)
        data = query_and_get_result(img, query_num, season_filter_data)
        res = {}
        res[output_col] = gr.Column(visible=True)
        for i in range(query_num):
            res[rows[i]['row']] = gr.Column(visible=True)
            if len(data) > i:
                ep = data[i]
                res[rows[i]['season_id']] = ep["season_id"]
                res[rows[i]['episode_id']] = ep["episode_id"]
                res[rows[i]['raw_time']] = ep["time"]
                res[rows[i]['video']] = None
                
                res[rows[i]['rank']] = i + 1
                res[rows[i]['season']] = f'第 {ep["season_id"]} 季：{ep["season_name"]}'
                res[rows[i]['episode']] = f'第 {ep["episode_id"]} 集：{ep["episode_name"]}'
                if ep['time'] < 60:
                    tim_str = f'{ep["time"]:.2f} 秒'
                elif ep['time'] < 3600:
                    tim_str = f'{int(ep["time"] // 60)} 分 {ep["time"] % 60:.2f} 秒'
                else:
                    tim_str = f'{int(ep["time"] // 3600)} 小时 {int(ep["time"] // 60 % 60)} 分 {ep["time"] % 60:.2f} 秒'
                res[rows[i]['time']] = tim_str
                res[rows[i]['sim']] = f'{ep["similarity"]:.2f} %'
        for i in range(query_num, max_textboxes):
            res[rows[i]['row']] = gr.Column(visible=False)
        return res
    
    output = [output_col]
    for i in range(max_textboxes):
        for key in rows[i]:
            output.append(rows[i][key])

    submit_btn.click(
        submit,
        [img, num_input, season_filter],
        output
    )

if __name__ == "__main__":
   demo.launch()