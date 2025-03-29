import cv2
import os
from tqdm import tqdm
import sys
sys.path.append(".")
import src.pose as pose
import src.object_detect as object_detect

def pose_from_frames(main_folder, sub_folder_name):

    input_folder = os.path.join(main_folder, "scene")
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    if not frame_files:
        print("Error: No frames found in the folder.")
        return
    
    with tqdm(total=len(frame_files), desc="Processing frames") as pbar:
        for index, frame_file in enumerate(frame_files):
            frame_path = os.path.join(input_folder, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                print(f"Warning: Could not read {frame_file}")
                continue
            
            # Detect pedestrians
            pose_result = pose.inference(frame)
            # Plot the detected poses
            # frame = pose_result.plot()
            # Get the cropped region of interest
            pose_crops = pose.get_crops(frame, pose_result)

            for ped_idx, pedestrian_crop in enumerate(pose_crops):

                sub_folder = f"{main_folder}/{sub_folder_name}{ped_idx}"
                # Create output folder if it doesn't exist
                os.makedirs(sub_folder, exist_ok=True)

                output_filename = os.path.join(sub_folder, f"frame_{index:04d}.png")
                cv2.imwrite(output_filename, pedestrian_crop)
                
            pbar.update(1)
            
if __name__ == "__main__":
    # Example usage
    input_folder = "/home/mi3/RPMS_Tonko/RMPS/data/sanity/input/video_0153"  # Change to your video file
    sub_folder_name = "pedestrian"   # Folder to save frames
    pose_from_frames(input_folder, sub_folder_name)
