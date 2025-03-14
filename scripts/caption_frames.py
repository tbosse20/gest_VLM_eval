import os
import pandas as pd
from tqdm import tqdm
import sys
sys.path.append(".")
from qwen import inference

csv_path = "data/sanity/output/caption_man_window=8_explain.csv"
columns = ["video_name", "frame_idx", "caption"]
# Generate csv file if not exists
if not os.path.exists(csv_path):
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, mode="w", index=False, header=True)

window = 8 # Batch size (<16)
for i in tqdm(range(0, 160 - window, window), desc="Processing"):
    respond = inference(
        video_folder="data/sanity/input/video_0153/pedestrian_man",
        start_frame=i,
        n_frames=window,
    )

    df = pd.DataFrame({"video_name": ['video_0153_man'], "frame_idx": [i], "caption": [respond]})
    df.to_csv(csv_path, mode="a", index=False, header=False)
