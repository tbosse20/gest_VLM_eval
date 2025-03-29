import cv2
import os
from tqdm import tqdm
from ultralytics import YOLO
import torch
import cv2
import logging
import numpy as np

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def extract_person_from_videos(videos_folder: str):
    """
    Extracts people from video folder and saves them with bounding boxes and tracking IDs in a CSV file.
    
    Args:
        videos_folder (str): Path to the folder containing video files.
    """
    
    # Check if the video folder exists and is a directory
    if not os.path.exists(videos_folder):
        raise FileNotFoundError(f"Main folder {videos_folder} does not exist.")
    if not os.path.isdir(videos_folder):
        raise NotADirectoryError(f"Main folder {videos_folder} is not a directory.")
    
    # Get videos from the main folder
    video_files = sorted([f for f in os.listdir(videos_folder) if f.endswith(('.mp4', '.avi', '.mov', '.MP4'))])
    if len(video_files) == 0:
        print("Error: No video files found in the folder.")
        return
    
    # Create output folder for labels
    OUTPUT_FOLDER = "data/labels/"
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
    # Make csv file for the video
    csv_file = os.path.basename(videos_folder) + "_bboxes.csv"
    csv_path = os.path.join(OUTPUT_FOLDER, csv_file)
    if os.path.exists(csv_path): os.remove(csv_path)
    with open(csv_path, 'w') as f: f.write("video_name,frame_id,pedestrian_id,x1,y1,x2,y2\n")
    
    # Process each video file
    for video_file in tqdm(video_files, desc="Videos"):
        video_path = os.path.join(videos_folder, video_file)
        pose_from_video(video_path, csv_path)

def add_tqdm(element, video_path):

    # Get total frame count using OpenCV
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    # Get video file name
    video_file = os.path.basename(video_path)
    
    # Initialize tqdm progress bar
    element = tqdm(element, total=total_frames, desc=video_file, unit="frames")
    
    return element

def pose_from_video(video_path: str, csv_path: str):
    # Load YOLO model
    yolo = YOLO("weights/yolov8s.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo.to(device).eval()

    # Track full video and stream frame-by-frame
    results = yolo.track(
        source=video_path,
        stream=True,
        persist=True,
        conf=0.1,
        iou=0.6,
    )

    results = add_tqdm(results, video_path)
    for frame_number, result in enumerate(results):
        if result.boxes is None:
            continue
        
        # Exclude non-person detections
        person_boxes = result.boxes[result.boxes.cls == 0]
        if len(person_boxes) == 0:
            continue
        
        # Get video file name and dimensions
        video_file = os.path.basename(result.path)
        width, height = result.orig_shape[1], result.orig_shape[0]

        for box in person_boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1_norm, y1_norm, x2_norm, y2_norm = np.array([x1, y1, x2, y2]) / np.array([width, height, width, height])
            track_id = int(box.id[0]) if box.id is not None else -1

            # Append result to CSV
            with open(csv_path, 'a') as f:
                f.write(f"{video_file},{frame_number},{track_id},{x1_norm},{y1_norm},{x2_norm},{y2_norm}\n")
                
if __name__ == "__main__":
    
    # Set the video folder and CSV file path
    videos_folder = "C:/Users/Tonko/OneDrive/Dokumenter/School/Merced/actedgestures"
    extract_person_from_videos(videos_folder)