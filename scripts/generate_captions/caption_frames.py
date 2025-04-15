import os
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from config.prompts import prompts
import models.src.utils as utils
import re

def caption_frames(video_path: str, window: int, interval: int, model_package = None, model_module = None):

    # Validate variables
    def raises(video_path: str, window: int):
        # Validate video path    
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path {video_path} not found")
        if not os.path.isdir(video_path):
            raise NotADirectoryError(f"Video path {video_path} is not a folder")
        
        # Ensure window is valid
        if window < 1:
            raise ValueError("Window must be greater than 0")
        if window > 16:
            raise ValueError("Window must be less than or equal to 16")
    raises(video_path, window)
    
    # Output csv path
    def output_csv(model_module):
        
        # Create csv file path
        OUTPUT_FOLDER_PATH = 'results/data/captions'
        if not os.path.exists(OUTPUT_FOLDER_PATH):
            os.makedirs(OUTPUT_FOLDER_PATH)

        # 
        module_name = model_module.__name__.split(".")[-1]
        csv_path = f"{OUTPUT_FOLDER_PATH}/{module_name}.csv"
        
        # Generate csv file if not exists
        columns = [
            "video_name",
            "frame_idx",
            "end_frame",
            "interval",
            'window_size',
            "prompt_type",
            "caption"
        ]
        
        # Generate csv file if not exists
        if not os.path.exists(csv_path):
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_path, mode="w", index=False, header=True)
        
        return csv_path
    csv_path = output_csv(model_module)
    
    # Get video name
    video_name = os.path.basename(video_path)
    
    # Skip computed videos
    computed_video_names = pd.read_csv(csv_path, index_col=False)["video_name"].values
    if video_name in computed_video_names:
        print(f"'{video_name}' already captioned, skip..")
        return
    
    # Get highest and lowest 0000 value in folder
    frame_idx = [int(frame.split("_")[-1].split(".")[0]) for frame in os.listdir(video_path)]
    start_frame, end_frame = min(frame_idx), max(frame_idx)

    # Iterate over video frames
    for i in tqdm(range(start_frame, end_frame - interval * window, interval * window), desc=f"{video_name}"):

        # Generate frames list
        frames_list = utils.generate_frame_list(video_path, i, interval, n_frames=window)
        if len(frames_list) == 0:     continue # Skip if no frames found
        if len(frames_list) < window: continue # Skip if less than window frames

        # Prepare dictionary
        dictionary = {
            "video_name":   [video_name],
            "start_frame":  [i],
            "end_frame":    [i + window],
            "interval":     [interval],
            "window_size":  [window],
        }
        
        # Iterate over prompts
        for prompt_type, prompt in prompts.items():
            
            # Get prompt
            dictionary['prompt_type'] = [prompt_type]
            
            # Get model response and append to dictionary
            respond = model_module.inference(prompt, frames_list, model_package)
            respond = re.sub(r' {2,}', '\\\\s', respond.replace('\n', '\\\\n').strip())
            dictionary['caption'] = [respond]

            # Save to csv
            df = pd.DataFrame(dictionary)
            df.to_csv(csv_path, mode="a", index=False, header=False)

def caption_folder(data_folder: str, window: int, interval: int, model_package, model_module):
    
    # Validate folder path
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} not found")
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"Data folder {data_folder} is not a folder")
    
    # Subfolders
    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
    
    for subfolder in subfolders:
        
        # Skip 'full_frame' folders
        if 'full_frame' in subfolder: continue
        
        # Caption frames
        caption_frames(subfolder, window, interval, model_package, model_module)

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Caption frames")
    parser.add_argument("video_folder", type=str, help="Video folder path")
    parser.add_argument("data_folder",  type=str, help="Data folder path")
    parser.add_argument("csv_path",     type=str, help="CSV path")
    parser.add_argument("window",       type=int, help="Window size", default=1)
    args = parser.parse_args()
    
    # Default values
    csv_path = "results/data/sanity/captions.csv" or args.csv_path
    
    if args.video_folder is None and args.data_folder is None:
        raise ValueError("Either video_folder or data_folder must be provided")
    
    # Load model
    import sys
    sys.path.append(".")
    import models.archive.qwen as model_module
    model_package = model_module.load_model()
    
    if args.video_folder is not None:
        caption_frames(args.video_folder, csv_path, args.window, model_package, model_module)
    
    if args.data_folder is not None:
        caption_folder(args.data_folder, csv_path, args.window, model_package, model_module)
        
    # Unload model
    utils.unload_model(*model_package)