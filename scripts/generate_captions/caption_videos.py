import importlib
import sys
import os
sys.path.append(".")
import models.src.utils as utils
from config.prompts import prompts
import pandas as pd
import re

def caption_videos(data_folder, models_folder):

    # Validate folder path
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} not found")
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"Data folder {data_folder} is not a folder")
    
    # Videos
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv', '.wmv')
    video_paths = [
        f.path
        for f in os.scandir(data_folder)
        if f.is_file() and f.path.lower().endswith(video_extensions)
    ]

    # Load all models modules
    model_modules = [
        f'{models_folder}.{module[:-3]}'
        for module in os.listdir(models_folder)
        if module.endswith(".py") and module != "__init__.py"
    ]

    # Iterate over models
    for name in model_modules:
        model_module = importlib.import_module(name)
        print('Processing using:', model_module.__name__.split(".")[-1])
        
        # Load model
        model_package = model_module.load_model()
        
        # Captions videos across prompts
        for video_path in video_paths:
            caption_video(video_path, model_package, model_module)
            
        # Unload model
        utils.unload_model(*model_package)
        
        # Delete model from system
        del model_module
        if name in sys.modules:
            del sys.modules[name]


def caption_video(video_path: str, model_package = None, model_module = None):

    # Validate variables
    def raises(video_path: str):
        # Validate video path    
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video path {video_path} not found")
    raises(video_path)
    
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
    
    # Iterate over prompts
    for prompt_type, prompt in prompts.items():
        
        # Get model response and append to dictionary
        respond = model_module.video_inference(prompt, video_path, model_package)
        respond = re.sub(r' {2,}', '\\\\s', respond.replace('\n', '\\\\n').strip())

        # Save to csv
        df = pd.DataFrame({
            "video_name":   [video_name],
            "prompt_type":  [prompt_type],
            "caption":      [respond],
        })
        df.to_csv(csv_path, mode="a", index=False, header=False)

if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser(description="Caption frames")
    # parser.add_argument("video_folder", type=str, help="Video folder path")
    # # parser.add_argument("data_folder",  type=str, help="Data folder path")
    # # parser.add_argument("csv_path",     type=str, help="CSV path")
    # parser.add_argument("window",       type=int, help="Window size", default=1)
    # args = parser.parse_args()

    # data_folder = "../realworldgestures_frames" # Path to video frames folder 
    data_folder = "/home/mi3/RPMS_Tonko/actedgestures" # Path to video frames folder 
    models_folder = "models"
    caption_videos(data_folder, models_folder)