# %%

import cv2
import pandas as pd
import os
import sys
sys.path.append(".")
from config.gesture_classes import Gesture

interval = 1 # Interval for frame extraction (in seconds)

def get_updated_csv(videos_folder_path):
    """ Get the updated CSV file path based on the video folder name. """
    
    OUTPUT_FOLDER = "data/labels/"
    if not os.path.exists(OUTPUT_FOLDER):
        raise FileNotFoundError(f"Output folder {OUTPUT_FOLDER} does not exist.")
    if not os.path.isdir(OUTPUT_FOLDER):
        raise NotADirectoryError(f"Output folder {OUTPUT_FOLDER} is not a directory.")
    
    # Get the base name of the video folder
    videos_folder_name = os.path.basename(os.path.normpath(videos_folder_path))
    # Check for the existence of the CSV file in this order
    CSV_TYPES = [
        "stretched",
        "bboxes"
    ]
    for csv_type in CSV_TYPES:
        print(f"Checking for {csv_type} CSV file...")
        csv_file = f"{videos_folder_name}_{csv_type}.csv"
        csv_path = os.path.join(OUTPUT_FOLDER, csv_file)

        if os.path.exists(csv_path):
            return csv_path
    
    raise FileNotFoundError(f"CSV file for {videos_folder_name} not found in {OUTPUT_FOLDER}.")

def visualize_results(videos_folder_path):

    # Check if the video folder and CSV file exist
    if not os.path.exists(videos_folder_path):
        raise FileNotFoundError(f"Video folder {videos_folder_path} does not exist.")
    if not os.path.isdir(videos_folder_path):
        raise NotADirectoryError(f"Video folder {videos_folder_path} is not a directory.")
    
    # Get the updated CSV file path
    csv_path = get_updated_csv(videos_folder_path)

    # Load the CSV file
    df = pd.read_csv(csv_path, index_col=False) if os.path.exists(csv_path) else None
    
    # Get videos by unique video names in the CSV file
    video_names = df["video_name"].unique() if df is not None else []
    if len(video_names) == 0:
        print("Error: No video names found in the CSV file.")
        return
    
    for video_name in video_names:
        video_path = os.path.join(videos_folder_path, video_name)
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            continue
        if not os.path.isfile(video_path):
            print(f"Error: Video file {video_path} is not a file.")
            continue
        if not video_path.endswith(('.mp4', '.avi', '.mov', '.MP4')):
            print(f"Error: Video file {video_path} is not a valid video format.")
            continue
        
        # Visualize the video with bounding boxes
        visualize_video(video_path, df)

def control_video_playback(play, frame_id, total_frames):
    """ Control video playback with keyboard input. """
    
    global interval
    
    # Get key press
    key = cv2.waitKeyEx(10) if play else cv2.waitKeyEx(0)
    
    # Control playback state
    play = not play if  key == 32   else play # Space to toggle play/pause
    exit() if           key == 113  else None # 'q' to exit
    
    # Control playback speed and frame navigation
    interval *= 2 if key == 2490368 else 1 # Up arrow
    interval /= 2 if key == 2621440 else 1 # Down arrow
    interval = int(max(1, interval)) # Ensure interval is at least 1
    
    frame_id += interval if play           else 0 # Play mode
    frame_id += interval if key == 2555904 else 0 # Right arrow
    frame_id -= interval if key == 2424832 else 0 # Left arrow
    
    # Keep frame_id within bounds by wrapping around
    frame_id = 0 if frame_id >= total_frames else frame_id
    frame_id = total_frames - 1 if frame_id < 0 else frame_id
    
    return play, frame_id

def draw_bounding_boxes(frame, df_video, frame_id, width, height):
    
    # Set the color and font for the bounding boxes
    COLOR = (0, 255, 0)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get the pedestrian IDs for the current frame
    pedestrian_ids = df_video[df_video["frame_id"] == frame_id]["pedestrian_id"]
    if len(pedestrian_ids) == 0:
        return frame
    
    # Check for duplicate pedestrian IDs
    if pedestrian_ids.duplicated().any():
        location = (width // 2 - 200, height // 2)
        cv2.putText(
            frame, f"Duplicate IDs detected\n{pedestrian_ids}", location,
            FONT, 1, (0, 0, 255), 2, cv2.LINE_AA
        )
    
    for pedestrian_id in pedestrian_ids:
        # Filter the DataFrame for the current pedestrian ID
        df_pedestrian = df_video[
            (df_video["pedestrian_id"] == pedestrian_id) & 
            (df_video["frame_id"] == frame_id)
        ]
        if df_pedestrian.empty:
            continue

        # Get the bounding box coordinates
        x1_norm, y1_norm, x2_norm, y2_norm = df_pedestrian.iloc[0][["x1", "y1", "x2", "y2"]].values     
        x1, y1, x2, y2 = x1_norm * width, y1_norm * height, x2_norm * width, y2_norm * height
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        string = f"ID: {str(pedestrian_id)}"
        location = (x1 + 5, y1 + 25)
        cv2.putText(frame, string, location, FONT, 0.7, COLOR, 2)
        
        # Draw the gesture ID if available
        gesture_label_id = (
            df_pedestrian.iloc[0]["gesture_label_id"]
            if "gesture_label_id" in df_pedestrian.columns
            else None
        )
        if gesture_label_id is not None:
            gesture_name = Gesture.get(gesture_label_id, "NaN")
            string = f"Gesture: {gesture_name} ({str(gesture_label_id)})"
            location = (x1 + 5, y1 + 50)
            cv2.putText(frame, string, location, FONT, 0.7, COLOR, 2)
    
    return frame

def visualize_video(video_path, df):
    global interval
    
    # Get the video name from the path
    video_name = os.path.basename(video_path)
    
    # Filter the DataFrame for the current video
    df_video = df[df["video_name"] == video_name]
    if df_video.empty:
        print(f"Error: No data found for video {video_name} in the CSV file.")
        return

    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize variables
    play = False
    frame_id = 0

    # Create a window to display the video
    while cap.isOpened():
        
        # Check if the video is playing or paused and read the frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: break
        
        # Draw bounding boxes on the frame
        frame = draw_bounding_boxes(frame, df_video, frame_id, width, height)
        cv2.putText(
            frame, f'Video: {video_name}', (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame, f'Frame: {frame_id}', (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            frame, f'Interval: {interval}', (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Processed Video', frame)
        play, frame_id = control_video_playback(play, frame_id, total_frames)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    # Add args
    import argparse
    parser = argparse.ArgumentParser(description="Extract people from videos and save to CSV.")
    parser.add_argument("--videos_folder",  type=str, help="Path to the folder containing video files.", required=True)
    args = parser.parse_args()
    
    visualize_results(args.videos_folder)