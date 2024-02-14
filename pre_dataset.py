import os
import cv2
import random
import argparse
from shutil import copyfile
import numpy as np

def list_video_files(directory):
    video_files = []
    for entry in os.scandir(directory):
        if entry.is_dir():
            video_files.extend(list_video_files(entry.path))
        elif entry.is_file() and entry.name.lower().endswith(('.mp4', '.avi', '.mkv')):
            video_files.append(entry.path)
    return video_files

def is_pure_color(frame, threshold=1):
    # 将图像转为灰度
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 计算颜色标准差
    std_dev = np.std(gray_frame)

    # 如果标准差低于阈值，认为是纯色
    return std_dev < threshold

def resize_frame(frame, target_long_side):
    # 获取原始图像的长宽
    h, w = frame.shape[:2]

    # 计算缩放比例
    scale = target_long_side / max(h, w)

    # 调整图像大小
    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    return resized_frame

train_num = 0
val_num = 0
def extract_frames(video_path, op=0, ed=0, num=1, output_path='output', split=0.2, quality=90, color_threshold=1, target_resolution=None):
    global train_num, val_num
    # 创建输出文件夹
    train_path = os.path.join(output_path, 'train')
    val_path = os.path.join(output_path, 'val')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # 获取所有视频文件
    video_files = list_video_files(video_path)

    print('Extracting frames from {} videos...'.format(len(video_files)))

    for video_file in video_files:
        # 生成输出文件名
        output_filename = os.path.splitext(os.path.basename(video_file))[0]

        # 打开视频文件
        cap = cv2.VideoCapture(video_file)

        # 获取视频帧率和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('Extracting frames from {}...'.format(output_filename))

        # 计算起始和结束帧
        start_frame = int(op * fps)
        end_frame = total_frames - int(ed * fps)

        # 等间隔选择帧
        interval = int((end_frame - start_frame) / (num + 1))
        selected_frames = [start_frame + i * interval for i in range(1, num + 1)]

        # 设置 cap 到指定位置
        for frame_pos in selected_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)

            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break

            

            # 调整图像大小
            if target_resolution:
                frame = resize_frame(frame, target_resolution)

            if is_pure_color(frame, color_threshold):
                continue
            
            # 决定是存储到 train 还是 val
            is_val = random.uniform(0, 1) < split

            if is_val:
                val_num += 1
                file_name = str(val_num) + '.jpg'
                output_folder = val_path
            else:
                train_num += 1
                file_name = str(train_num) + '.jpg'
                output_folder = train_path

            # 生成输出文件路径
            output_filepath = os.path.join(output_folder, file_name)

            # 保存帧
            cv2.imwrite(output_filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        # 释放视频捕获对象
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos and save to train/val folders.")
    parser.add_argument("-video_path", default='videos', help="Path to the video folder (default: videos).")
    parser.add_argument("-op", type=int, default=0, help="Length of the video header to be ignored (default: 0).")
    parser.add_argument("-ed", type=int, default=0, help="Length of the video tail to be ignored (default: 0).")
    parser.add_argument("-num", type=int, default=10, help="Number of frames to extract from each video (default: 10).")
    parser.add_argument("-dataset", default='dataset', help="Output dataset folder for the frames (default: dataset).")
    parser.add_argument("-split", type=float, default=0.2, help="Validation set split ratio (default: 0.2).")
    parser.add_argument("-quality", type=int, default=90, help="Quality of the saved images (default: 90).")
    parser.add_argument("-color_threshold", type=float, default=1, help="Threshold for detecting pure color frames (default: 1).")
    parser.add_argument("-target_resolution", type=int, help="Target resolution for the long side of the images (default: None).")

    args = parser.parse_args()

    extract_frames(args.video_path, args.op, args.ed, args.num, args.output_path, args.split, args.quality, args.color_threshold, args.target_resolution)
