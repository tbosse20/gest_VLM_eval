import os
import cv2
import torch
import numpy as np

def from_end_frame(video_folder, start_frame, interval, end_frame):
    return [
        f"{video_folder}/frame_{frame_count:04d}.png"
        for frame_count in range(start_frame, end_frame, interval)
    ]

def from_n_frame(video_folder, start_frame, interval, n_frames):
    return [
        f"{video_folder}/frame_{start_frame + frame_count:04d}.png"
        for frame_count in range(0, n_frames * interval, interval)
        if os.path.exists(f"{video_folder}/frame_{start_frame + frame_count:04d}.png")
    ]

def generate_frame_list(video_folder, start_frame=0, interval=1, end_frame=None, n_frames=None):
    
    # Validate video folder
    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder {video_folder} not found")
    if not os.path.isdir(video_folder):
        raise NotADirectoryError(f"Video folder {video_folder} is not a folder")
    
    # Get the highest frame number, if no end- or n-frames are provided
    if end_frame is None and n_frames is None:
        highest_frame = max([int(frame.split("_")[-1].split(".")[0]) for frame in os.listdir(video_folder)])
        end_frame = highest_frame + 1
    
    # Generate the frame list
    if end_frame is not None: 
        return from_end_frame(video_folder, start_frame, interval, end_frame)
    if n_frames is not None: 
        return from_n_frame(video_folder, start_frame, interval, n_frames)

# Function to create a video from images
def create_video_from_frames(frames: list[np.ndarray], output_video_path):

    # Get frame size
    height, width, _ = frames[0].shape
    size = (width, height)

    # Create video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 1, size)

    for img in frames:
        out.write(img)

    out.release()

def create_video_from_str(frame_paths: list[str], output_video_path):
    
    # Set
    FRAMES_FOLDER = '/home/mi3/RPMS_Tonko/RMPS'

    # Load frames
    frames = [
        cv2.imread(f'{FRAMES_FOLDER}/{frame_path}')
        for frame_path in frame_paths
        if os.path.exists(frame_path)
    ]

    # Create video
    create_video_from_frames(frames, output_video_path)

def unload_model(*args):
    for obj in args:
        del obj
        obj = None
    
    torch.cuda.empty_cache()
    
def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Generate video captions using Qwen2VL model.")
    parser.add_argument("--video_folder", type=str, help="Path to the video folder containing frames.", required=True)
    parser.add_argument("--prompt",       type=str, help="The prompt to generate captions.",            default='')
    parser.add_argument("--start_frame",  type=int, help="The starting frame number.",                  default=0)
    parser.add_argument("--interval",     type=int, help="The interval between frames.",                default=1)
    parser.add_argument("--end_frame",    type=int, help="The ending frame number.",                    default=None)
    parser.add_argument("--n_frames",     type=int, help="The number of frames to process.",            default=None)
    
    return parser.parse_args()

if __name__ == "__main__":
    
    for i in range(0, 160, 8):
        frame_list = generate_frame_list(
            start_frame = i,
            interval = 1,
            n_frames = 8
        )
        
    frame_list = generate_frame_list(
        "../video_frames/Go forward",
        n_frames = 8
    )
    
    print(frame_list)