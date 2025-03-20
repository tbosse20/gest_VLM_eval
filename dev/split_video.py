import cv2
import os
from tqdm import tqdm

def save_video_frames(video_path, output_folder, interval: int = 1):
    
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
            
            # Save frame in interval
            if frame_idx % interval == 0:
                frame_filename = os.path.join(output_folder, f"frame_{frame_idx:04d}.png")
                cv2.imwrite(frame_filename, frame)
            
            # Update progress bar
            pbar.update(1)
            frame_idx += 1
    
    cap.release()
    
def save_video_frames_folder(videos_folder: str, interval: int):
    
    # Validate input folder
    if not os.path.exists(videos_folder):
        raise FileNotFoundError(f"Input folder '{videos_folder}' not found.")
    if not os.path.isdir(videos_folder):
        raise NotADirectoryError(f"Input folder '{videos_folder}' is not a directory")
    
    # Create output folder if it doesn't exist
    output_folder = videos_folder + "_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all video names in the folder
    video_names = os.listdir(videos_folder)
    for video_name in video_names:
        # 
        if not video_name.endswith(".mp4"): continue
        # Get video path
        video_path = f"{videos_folder}/{video_name}"
        # Create output folder for video frames
        video_output_folder = f"{output_folder}/{video_name.split('.')[0]}"
        # Create output folder if it doesn't exist
        save_video_frames(video_path, video_output_folder, interval)

if __name__ == "__main__":
    
    videos_folder = "../realworldgestures"  # Folder containing videos
    interval = 18 # 36 fps / 18 = 2 fps
    save_video_frames_folder(videos_folder, interval)