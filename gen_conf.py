import os
import json
import argparse
import cv2

def parse_episode_name(episode_file):
    separators = [' ', '_', '-']
    
    for separator in separators:
        episode_info = episode_file.split(separator, 1)
        if episode_info[0].isdigit() and len(episode_info) == 2:
            return {"id": int(episode_info[0]), "name": episode_info[1].split(".")[0]}

    # If no valid separator is found, return None
    return None

def get_video_info(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frames per second (fps) and calculate duration
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    # Release the video capture object
    cap.release()

    return duration, fps

def generate_filelist(video_path, input=None, output="conf.json", output_diff="conf_diff.json", fps=8, op_length=0, ed_length=0, full_first_ep=True, sim_threshold=4):

    if input is not None:
        with open(input, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        settings = input_data["settings"]
        lst_season_id = input_data["seasons"][-1]['id']
        lst_episode_id = input_data["seasons"][-1]['episodes'][-1]['id']
        index_id = input_data["seasons"][-1]['index_range'][1] + 1
        fps = settings["fps"]
    else:
        lst_season_id = -1
        lst_episode_id = -1
        index_id = 0
    seasons = []

    for season_folder in os.listdir(video_path):
        season_path = os.path.join(video_path, season_folder)
        if os.path.isdir(season_path):
            season_info = parse_episode_name(season_folder)
            
            if season_info:
                season_id = season_info["id"]
                if season_id < lst_season_id:
                    continue
                season_name = season_info["name"]

                episodes = []
                for episode_file in os.listdir(season_path):
                    episode_path = os.path.join(season_path, episode_file)
                    if os.path.isfile(episode_path):
                        episode_info = parse_episode_name(episode_file)
                        if episode_info:
                            episode_id = episode_info["id"]
                            if season_id == lst_season_id and episode_id <= lst_episode_id:
                                continue
                            episode_name = episode_info["name"]
                            episodes.append({"name": episode_name, "id": episode_id, "path": episode_path})
                episodes.sort(key=lambda x: x["id"])
                if len(episodes) > 0:
                    seasons.append({"name": season_name, "id": season_id, "episodes": episodes})

    
    seasons.sort(key=lambda x: x["id"])

    for season in seasons:
        for episode in season["episodes"]:
            episode_duration, vfps = get_video_info(episode["path"])
            episode["fps"] = vfps
            episode["duration"] = episode_duration
            episode_index_start = index_id
            index_id += int(episode_duration * fps) + 1
            episode_index_end = index_id - 1
            episode["index_range"] = [episode_index_start, episode_index_end]
        season_index_start = season["episodes"][0]["index_range"][0] if season["episodes"] else 0
        season_index_end = season["episodes"][-1]["index_range"][1] if season["episodes"] else 0
        season["index_range"] = [season_index_start, season_index_end]
    
    settings = {
        "fps": fps,
        "op_length": op_length,
        "ed_length": ed_length,
        "full_first_ep": full_first_ep,
        "sim_threshold": sim_threshold
    }
    
    if input is None or len(seasons) == 0:
        filelist = {"settings": settings, "seasons": seasons}
        diff_list = {"settings": settings, "seasons": seasons}
    else:
        pre_seasons = input_data["seasons"]
        if pre_seasons[-1]['id'] == seasons[0]['id']:
            pre_seasons[-1]['episodes'].extend(seasons[0]["episodes"])
            pre_seasons[-1]['index_range'][1] = seasons[0]["index_range"][1]
            pre_seasons.extend(seasons[1:])
        else:
            pre_seasons.extend(seasons)
        filelist = {"settings": settings, "seasons": pre_seasons}
        diff_list = {"settings": settings, "seasons": seasons}
    with open(output, 'w', encoding="utf-8") as file:
        json.dump(filelist, file, indent=2, ensure_ascii=False)
    with open(output_diff, 'w', encoding="utf-8") as file:
        json.dump(diff_list, file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON file list for video episodes.")
    parser.add_argument("-video_path", default='videos', help="Path to the video files folder (default: videos)")
    parser.add_argument("-input", default=None, help="Input configuration file path (default: None)")
    parser.add_argument("-output_diff", default="conf_diff.json", help="Output difference file path (default: conf_diff.json)")
    parser.add_argument("-output", default="conf.json", help="Output file path (default: conf.json)")
    parser.add_argument("-fps", type=int, default=8, help="Frames per second for image extraction (default: 8)")
    parser.add_argument("-op", type=int, default=0, help="OP length in seconds (default: 0)")
    parser.add_argument("-ed", type=int, default=0, help="ED length in seconds (default: 0)")
    parser.add_argument("-full_first_ep", type=bool, default=True, help="Include OP/ED in the processing of the first episode (default: True)")
    parser.add_argument("-sim_threshold", type=int, default=4, help="Similarity threshold for image extraction (default: 4)")
    args = parser.parse_args()

    generate_filelist(args.video_path, args.input, args.output, args.output_diff, args.fps, args.op, args.ed, args.full_first_ep, args.sim_threshold)
