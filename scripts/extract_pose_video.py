import cv2
import os
from tqdm import tqdm
import sys
sys.path.append(".")
from ultralytics import YOLO
import torch
import cv2
import logging
import numpy as np

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)
    
# --- Load YOLOv8 Model ---
yolo = YOLO("weights/yolov8s.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo.to(device).eval()

def pose_from_videos(videos_folder: str, csv_file: str):
    """
    Extracts pose from video frames and saves them in a specified folder.
    
    Args:
        main_folder (str): Path to the main folder containing the video frames.
        csv_file (str): Path to the CSV file where the pose data will be saved.
    """
    
    if not os.path.exists(videos_folder):
        raise FileNotFoundError(f"Main folder {videos_folder} does not exist.")
    if not os.path.isdir(videos_folder):
        raise NotADirectoryError(f"Main folder {videos_folder} is not a directory.")
    
    # Get videos from the main folder
    video_files = sorted([f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov'))])
    if len(video_files) == 0:
        print("Error: No video files found in the folder.")
        return
    
    # Make csv file for the video
    if os.path.exists(csv_file): os.remove(csv_file)
    with open(csv_file, 'w') as f: f.write("video_name,frame_id,pedestrian_id,x1,y1,x2,y2\n")
    
    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        pose_from_video(video_path, csv_file)
        
def pose_from_video(video_path: str, csv_file: str):
    
    # Check if the video file exists and is a file
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file {video_path} does not exist.")
    if not os.path.isfile(video_path):
        raise NotADirectoryError(f"Video file {video_path} is not a file.")
    
    # Load the video
    video_file = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    
    # Initialize progress bar
    pbar = tqdm(total=frame_count, desc=f"Proc. {video_file}")
    
    # Process video frames
    while cap.isOpened():
        
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret: break
        
        # Run YOLOv8 with tracking on current frame
        with torch.no_grad():
            results = yolo.track(
                source=frame,
                persist=True,   # keep tracking across frames
                stream=True,
                conf=0.1,       # confidence threshold
                iou=0.6,        # IoU threshold for NMS
            )
            
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        process_results(results, video_file, frame_number, width, height, csv_file)
        pbar.update(1)
            
def process_results(results, video_file, frame_number, width, height, csv_file):
    
    # Since 'results' is a generator, loop through it
    for pedestrian_result in results:
        
        # Filter only 'person' class (COCO class 0)
        if pedestrian_result.boxes is None: continue
        
        # Filter out non-person classes
        person_boxes = pedestrian_result.boxes[pedestrian_result.boxes.cls == 0]
        if len(person_boxes) == 0: continue
        
        # Process each detected pedestrian
        process_person(person_boxes, video_file, frame_number, width, height, csv_file)

def process_person(person_boxes, video_file, frame_number, width, height, csv_file):

    # Iterate through each detected pedestrian
    for ped_idx, pedestrian_crop in enumerate(person_boxes):
        x1, y1, x2, y2 = pedestrian_crop.xyxy[0].cpu().numpy()
        x1_norm, y1_norm, x2_norm, y2_norm = np.array([x1, y1, x2, y2]) / np.array([width, height, width, height])

        # Optional: get tracking ID
        track_id = int(pedestrian_crop.id[0]) if pedestrian_crop.id is not None else -1

        # Save to CSV
        with open(csv_file, 'a') as f:
            bbox_str = f"{x1_norm},{y1_norm},{x2_norm},{y2_norm}"
            f.write(f"{video_file},{frame_number},{track_id},{bbox_str}\n")
            
if __name__ == "__main__":
    
    # Example usage
    videos_folder = "data/videos/"
    csv_file = "data/labels/annotations.csv"
    pose_from_videos(videos_folder, csv_file)
