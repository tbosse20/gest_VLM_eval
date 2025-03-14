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
    
    # Rename csv_path
    csv_path = csv_path.replace(".csv", f"_window={window}_explain.csv")
    
    # Generate csv file if not exists
    columns = ["video_name", "frame_idx", "caption"]
    
    # Generate csv file if not exists
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, mode="w", index=False, header=True)

    # Load model
    model_package = load_model()

    # Iterate over video frames
    for i in tqdm(range(0, 160 - window, window), desc="Processing"):
        
        # Generate frames list
        frames_list = generate_frame_list(video_path, i, interval=1, n_frames=window)
        
        # Run inference
        respond = inference(
            prompt="explain the video",
            frames_list=frames_list,
        )

        # Save to csv
        df = pd.DataFrame({
            "video_name":   ['video_0153_man'],
            "frame_idx":    [i],
            "caption":      [respond]
        })
        df.to_csv(csv_path, mode="a", index=False, header=False)

    # Unload model
    unload_model(*model_package)

if __name__ == "__main__":
    video_folder = "data/sanity/input/video_0153"
    csv_path = "data/results/data/caption.csv"
    caption_frames(video_folder, csv_path, window=8)