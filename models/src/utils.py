import os
import cv2
import torch
import numpy as np
import sys
sys.path.append(".")
import enhance.augment.augment as augment
import enhance.video_pipeline as video_pipeline
import config.flags as flags

def from_end_frame(video_folder, start_frame, interval, end_frame):
    return [
        f"{video_folder}/frame_{frame_count:04d}.png"
        for frame_count in range(start_frame, end_frame, interval)
    ]

def from_n_frame(video_folder, start_frame, interval, n_frames):
    # Ensure we only generate frames based on the interval and n_frames
    return [
        f"{video_folder}/frame_{start_frame + frame_count * interval:04d}.png"
        for frame_count in range(n_frames)
        if os.path.exists(f"{video_folder}/frame_{start_frame + frame_count * interval:04d}.png")
    ]

def get_start_n_end_frames(video_path):
        
    # Get highest and lowest 0000 value in folder
    frame_indices = [
        int(frame.split("_")[-1].split(".")[0])
        for frame in os.listdir(video_path)
    ]
    return min(frame_indices), max(frame_indices)

def generate_frame_list(video_folder, start_frame=None, interval=1, end_frame=None, n_frames=None):
    
    # Validate video folder
    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"Video folder {video_folder} not found")
    if not os.path.isdir(video_folder):
        raise NotADirectoryError(f"Video folder {video_folder} is not a folder")
    
    # Automatically get lowest and/or highest frame as start- and end frame
    if not start_frame or (not end_frame and not n_frames):
        min_frame, max_frame = get_start_n_end_frames(video_folder)
        
        # Get lowest frame index, at the start frame
        start_frame = start_frame or min_frame
        
        # Get the highest frame index, if no end- or n-frames are provided
        end_frame = end_frame or (n_frames or max_frame + 1)
    
    # Generate the frame list
    from_method = from_end_frame if end_frame else from_n_frame
    frame_args = end_frame if end_frame else n_frames
    frame_list = from_method(video_folder, start_frame, interval, frame_args)

    if len(frame_list) == 0:
        raise (f"Frames are empty.")
    
    return frame_list

# Function to create a video from images
def create_video_from_frames(frames: list[np.ndarray], output_video_path):

    # Get frame size
    height, width, _ = frames[0].shape
    size = (width, height)

    temp_path = output_video_path.replace(".mp4", "_temp.mp4")

    # Write raw frames to a temp file
    out = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, size)
    for img in frames:
        out.write(img)
    out.release()

    # If enhancement is needed, process and overwrite output_video_path
    if flags.projection_enhancement:
        video_pipeline.from_video(
            method=augment.process_frame,
            video_path=temp_path,
            video_output=output_video_path,
            draw=1
        )
        os.remove(temp_path)  # Clean up the temp file
    else:
        os.replace(temp_path, output_video_path)  # Rename to final if no processing

def create_video_from_str(frame_paths: list[str]):
    
    # Set running directory
    FRAMES_FOLDER = os.path.abspath(__file__)
    TMP_FILE_PATH = f"_tmp_output.mp4"

    # Load frames
    frames = [
        cv2.imread(frame_path)
        for frame_path in frame_paths
        if os.path.exists(frame_path)
    ]

    # Create video
    create_video_from_frames(frames, TMP_FILE_PATH)
    
    return TMP_FILE_PATH

def unload_model(*args):
    for obj in args:
        del obj
        obj = None
    
    torch.cuda.empty_cache()

def argparse():
    import argparse

    parser = argparse.ArgumentParser(description="Generate video captions using selected VLM.")
    parser.add_argument("--video_folder", type=str, help="Path to the video folder containing frames.", required=True)
    parser.add_argument("--prompt",       type=str, help="The prompt to generate captions.",            default='')
    parser.add_argument("--start_frame",  type=int, help="The starting frame number.",                  default=0)
    parser.add_argument("--interval",     type=int, help="The interval between frames.",                default=1)
    parser.add_argument("--end_frame",    type=int, help="The ending frame number.",                    default=None)
    parser.add_argument("--n_frames",     type=int, help="The number of frames to process.",            default=None)
    args = parser.parse_args()

    if not os.path.exists(args.video_folder):
        raise ("Input does not exist.")

    if os.path.isdir(args.video_folder):
        frame_list = generate_frame_list(
            args.video_folder,
            args.start_frame,
            args.interval,
            args.end_frame,
            args.n_frames
        )
        return args.prompt, frame_list
    
    return args.prompt, args.video_folder

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