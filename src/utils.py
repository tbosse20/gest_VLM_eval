import os

def from_end_frame(video_folder, start_frame, interval, end_frame):
    return [
        f"{video_folder}/frame_{frame_count:04d}.png"
        for frame_count in range(start_frame, end_frame, interval)
    ]

def from_n_frame(video_folder, start_frame, interval, n_frames):
    return [
        f"{video_folder}/frame_{start_frame + frame_count:04d}.png"
        for frame_count in range(0, n_frames, interval)
        if os.path.exists(f"{video_folder}/frame_{start_frame + frame_count:04d}.png")
    ]

def generate_frame_list(video_folder, start_frame, interval=1, end_frame=None, n_frames=None):
    
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
