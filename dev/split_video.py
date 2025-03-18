import cv2
import os
from tqdm import tqdm

def save_video_frames(video_path, output_folder):
    
    # Create output folder if it doesn't exist
    if os.path.exists(output_folder):
        return
    else:
        os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get total number of frames
    frame_idx = 0
    # Get video name
    video_name = output_folder.split("/")[-1]
    
    # Loop through each frame
    with tqdm(total=frame_count, desc=f"Proc. {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Save frame
            frame_filename = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
            cv2.imwrite(frame_filename, frame)
            
            # Update progress bar
            pbar.update(1)
            frame_idx += 1
    
    cap.release()
    
def save_video_frames_folder(videos_folder: str, output_folder: str):
    
    # Validate input folder
    if not os.path.exists(videos_folder):
        raise FileNotFoundError(f"Input folder '{videos_folder}' not found.")
    if not os.path.isdir(videos_folder):
        raise NotADirectoryError(f"Input folder '{videos_folder}' is not a directory")
    
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Get all video names in the folder
    video_names = os.listdir(videos_folder)
    
    # Loop through each video
    for video_name in video_names:
        # Get video path
        video_path = f"{videos_folder}/{video_name}"
        # Create output folder for video frames
        video_output_folder = f"{output_folder}/{video_name.split('.')[0]}"
        # Create output folder if it doesn't exist
        save_video_frames(video_path, video_output_folder)

if __name__ == "__main__":
    
    videos_folder = "../realworldgestures_front"  # Folder containing videos
    # output_folder = "data/video_frames" # Folder to save frames
    output_folder = "../realworldgestures_video_frames" # Folder to save frames
    save_video_frames_folder(videos_folder, output_folder)