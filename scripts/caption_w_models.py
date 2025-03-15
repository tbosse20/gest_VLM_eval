import importlib
import sys
import os
sys.path.append(".")
import scripts.caption_frames as caption_frames 
import src.utils as utils

def caption_across_models(data_folder, models_folder, window):

    # Load all models modules
    model_modules = [
        f'{models_folder}.{module[:-3]}'
        for module in os.listdir(models_folder)
        if module.endswith(".py") and module != "__init__.py"
    ]

    # Iterate over models
    for name in model_modules:
        model_module = importlib.import_module(name)
        print('Processing using:', model_module.__name__)
        
        # Load model
        model_package = model_module.load_model()

        # Caption frames from all videos in folder
        caption_frames.caption_folder(data_folder, window, model_package, model_module)
            
        # Unload model
        utils.unload_model(*model_package)
        
        # Delete model from system
        del model_module
        if name in sys.modules:
            del sys.modules[name]

if __name__ == "__main__":

    # import argparse
    # parser = argparse.ArgumentParser(description="Caption frames")
    # parser.add_argument("video_folder", type=str, help="Video folder path")
    # parser.add_argument("data_folder",  type=str, help="Data folder path")
    # parser.add_argument("csv_path",     type=str, help="CSV path")
    # parser.add_argument("window",       type=int, help="Window size", default=1)
    # args = parser.parse_args()

    # data_folder = "data/video_frames" # Path to folder containing video frames
    data_folder = "/media/mi3/3562-6238/DCIM/video_frames" # Can't find SD
    data_folder = "../video_frames" # Path to folder containing video frames
    models_folder = "models"
    window = 8 # Number of frames to process at once
    caption_across_models(data_folder, models_folder, window)