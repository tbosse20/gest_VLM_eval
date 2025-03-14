import cv2
import os
from tqdm import tqdm

def save_video_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    frame_count = 0
    
    with tqdm(total=frame_count, desc="Processing frames") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Break when video ends
            
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            
            pbar.update(1)
            frame_count += 1
    
    cap.release()
    print("All frames have been saved.")

# Example usage
video_path = "data/sanity/input/video_0153.mp4"  # Change to your video file
output_folder = "data/sanity/input/video_0153"   # Folder to save frames
save_video_frames(video_path, output_folder)
