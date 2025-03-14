import importlib
import sys
import os
sys.path.append(".")
import scripts.caption_frames as caption_frames 
import src.utils as utils

# Constants
models_folder = "models"
# Default values (parsed into csv name)
data_folder = "data/video_frames"               # Path to folder containing video frames
window = 8                                      # Number of frames to process at once
csv_path = "results/data/sanity_captions.csv"   # Path to save captions

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
    caption_frames.caption_folder(data_folder, csv_path, window, model_package, model_module)
        
    # Unload model
    utils.unload_model(*model_package)
    
    # Delete model from system
    del model_module
    if name in sys.modules:
        del sys.modules[name]