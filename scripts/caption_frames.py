import os
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from config.prompts import prompts
import src.utils as utils

def caption_frames(video_path: str, window: int, model_package, model_module):
    
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

    # Create csv file path
    OUTPUT_FOLDER_PATH = 'results/data/captions'
    module_name = model_module.__name__.split(".")[-1]
    csv_path = f"{OUTPUT_FOLDER_PATH}/{module_name}.csv"
    
    # Generate csv file if not exists
    columns = ["video_name", "frame_idx"]
    columns += [prompt['alias'] for prompt in prompts]
    
    # Generate csv file if not exists
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, mode="w", index=False, header=True)
    
    # Get video name
    video_name = os.path.basename(video_path)

    # Iterate over video frames
    for i in tqdm(range(0, 160 - window, window), desc="Processing"):
        
        # Generate frames list
        frames_list = utils.generate_frame_list(video_path, i, interval=1, n_frames=window)
        
        # Prepare dictionary
        dictionary = {
            "video_name": [video_name],
            "frame_idx":  [i],
        }
        
        # Iterate over prompts
        for prompt in prompts:
            respond = model_module.inference(
                prompt=prompt['text'],
                frames_list=frames_list,
                model_package=model_package
            )
            dictionary[prompt['alias']] = [respond]

        df = pd.DataFrame(dictionary)
        df.to_csv(csv_path, mode="a", index=False, header=False)

def caption_folder(data_folder: str, window: int, model_package, model_module):
    
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
        caption_frames(subfolder, window, model_package, model_module)

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
    import models.qwen as model_module
    model_package = model_module.load_model()
    
    if args.video_folder is not None:
        caption_frames(args.video_folder, csv_path, args.window, model_package, model_module)
    
    if args.data_folder is not None:
        caption_folder(args.data_folder, csv_path, args.window, model_package, model_module)
        
    # Unload model
    utils.unload_model(*model_package)