import os
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from models.qwen import generate_frame_list, inference, load_model, unload_model

def caption_frames(video_path: str, csv_path: str, window: int < 16): # type: ignore
    
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

    # Load model
    model_package = load_model()
    model_name = model_package[0].__class__.__name__
    
    # Get prompt
    prompt = prompt[0]
    
    # Rename csv_path
    csv_path = csv_path.replace(".csv", f"_window={window}_explain.csv")
    csv_path = csv_path.replace(".csv", f"_{prompt.alias}.csv")
    csv_path = csv_path.replace(".csv", f"_model={model_name}.csv")
    
    # Generate csv file if not exists
    columns = ["video_name", "frame_idx", "caption"]
    
    # Generate csv file if not exists
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, mode="w", index=False, header=True)
    
    # Get video name
    video_name = os.path.basename(video_path)

    # Iterate over video frames
    for i in tqdm(range(0, 160 - window, window), desc="Processing"):
        
        # Generate frames list
        frames_list = generate_frame_list(video_path, i, interval=1, n_frames=window)
        
        # Run inference
        respond = inference(
            prompt=prompt.text,
            frames_list=frames_list,
        )

        # Save to csv
        df = pd.DataFrame({
            "video_name":   [video_name],
            "frame_idx":    [i],
            "caption":      [respond]
        })
        df.to_csv(csv_path, mode="a", index=False, header=False)

    # Unload model
    unload_model(*model_package)

def caption_folder(data_folder: str, csv_path: str, window: int < 16): # type: ignore
    
    # Validate folder path
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} not found")
    if not os.path.isdir(data_folder):
        raise NotADirectoryError(f"Data folder {data_folder} is not a folder")
    
    # Subfolders
    subfolders = [f.path for f in os.scandir(data_folder) if f.is_dir()]
    
    for subfolder in subfolders:
        caption_frames.caption_frames(subfolder, csv_path, window)

if __name__ == "__main__":
    # Argsparser
    import argparse
    parser = argparse.ArgumentParser(description="Caption frames")
    parser.add_argument("video_folder", type=str, help="Video folder path")
    parser.add_argument("data_folder",  type=str, help="Data folder path")
    parser.add_argument("csv_path",     type=str, help="CSV path")
    parser.add_argument("window",       type=int, help="Window size", default=1)
    args = parser.parse_args()
    
    # Default values
    csv_path = "data/sanity/output/captions.csv" or args.csv_path
    
    if args.video_folder is None and args.data_folder is None:
        raise ValueError("Either video_folder or data_folder must be provided")
    
    if args.video_folder is not None:
        caption_frames(args.video_folder, csv_path, args.window)
    
    if args.data_folder is not None:
        caption_folder(args.data_folder, csv_path, args.window)