# %%
import numpy as np
import cv2
import pandas as pd
import os
import sys
import random
sys.path.append(".")
from config.gesture_classes import Gesture

interval = 1 # Interval for frame extraction (in seconds)
show_hud = True # Show HUD (Heads-Up Display) with video name and frame number

def get_updated_csv(videos_folder_path, csv_type: str):
    """ Get the updated CSV file path based on the video folder name. """
    
    OUTPUT_FOLDER = "data/labels/"
    if not os.path.exists(OUTPUT_FOLDER):
        raise FileNotFoundError(f"Output folder {OUTPUT_FOLDER} does not exist.")
    if not os.path.isdir(OUTPUT_FOLDER):
        raise NotADirectoryError(f"Output folder {OUTPUT_FOLDER} is not a directory.")
    
    # Get the base name of the video folder
    videos_folder_name = os.path.basename(os.path.normpath(videos_folder_path))
    # Check for the existence of the CSV file in this order
    
    csv_file = f"{videos_folder_name}_{csv_type}.csv"
    csv_path = os.path.join(OUTPUT_FOLDER, csv_file)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=False)
        return df
    
    # raise FileNotFoundError(f"CSV file for {videos_folder_name} not found in {OUTPUT_FOLDER}.")

def visualize_results(videos_folder_path):

    # Check if the video folder and CSV file exist
    if not os.path.exists(videos_folder_path):
        raise FileNotFoundError(f"Video folder {videos_folder_path} does not exist.")
    if not os.path.isdir(videos_folder_path):
        raise NotADirectoryError(f"Video folder {videos_folder_path} is not a directory.")
    
    # Get the updated CSV file path
    df_bboxes = get_updated_csv(videos_folder_path, 'bboxes')
    df_labels = get_updated_csv(videos_folder_path, 'sequential')

    # Get videos by unique video names in the CSV file
    video_names = df_bboxes["video_name"].unique() if df_bboxes is not None else []
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

        df_video_labels = df_labels[df_labels["video_name"] == video_name.lower()]
        
        # Visualize the video with bounding boxes
        visualize_video(video_path, df_bboxes, df_video_labels)

def control_video_playback(play, frame_id, total_frames):
    """ Control video playback with keyboard input. """
    
    global interval
    global show_hud
    
    # Get key press
    key = cv2.waitKeyEx(10) if play else cv2.waitKeyEx(0)
    # print(f"Key pressed: {key}")
    
    # Control HUD visibility
    show_hud = not show_hud if key == 104 else show_hud # 'h' to toggle HUD
    
    # Control playback state
    play = not play if  key == 32   else play # Space to toggle play/pause
    exit() if           key == 113  else None # 'q' to exit
    
    # Control playback speed and frame navigation
    interval *= 2 if key == 2490368 else 1 # Up arrow
    interval /= 2 if key == 2621440 else 1 # Down arrow
    interval = int(max(1, interval)) # Ensure interval is at least 1
    
    # Control frame navigation
    frame_id += interval if play           else 0 # Play mode
    frame_id += interval if key == 2555904 else 0 # Right arrow
    frame_id -= interval if key == 2424832 else 0 # Left arrow
    
    # Keep frame_id within bounds by wrapping around
    frame_id = 0 if frame_id >= total_frames else frame_id
    frame_id = total_frames - 1 if frame_id < 0 else frame_id
    
    return play, frame_id

def get_pedestrian(df_video, frame_id, pedestrian_id):
    """ Get the pedestrian ID from the DataFrame. """
    
    # Filter the DataFrame for the current pedestrian ID
    df_pedestrian = df_video[
        (df_video["pedestrian_id"] == pedestrian_id) & 
        (df_video["frame_id"] == frame_id)
    ]
    
    # Check if the DataFrame is empty
    if df_pedestrian.empty:
        return None
    
    return df_pedestrian

def get_bbox_from_id(df_pedestrian, width, height):
    """ Get the bounding box coordinates for a given pedestrian ID. """

    # Get the bounding box coordinates
    x1_norm, y1_norm, x2_norm, y2_norm = df_pedestrian.iloc[0][["x1", "y1", "x2", "y2"]].values     
    x1, y1, x2, y2 = x1_norm * width, y1_norm * height, x2_norm * width, y2_norm * height
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    return x1, y1, x2, y2

def draw_bbox_id(frame, x1, y1, x2, y2, pedestrian_id):
    """ Draw the bounding box ID on the frame. """
    
    # Set the color and font for the bounding boxes
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SIZE = 0.5
    WIDTH = 1
    
    # Set the color and font for the bounding boxes
    # random.seed(pedestrian_id)
    # color = tuple(random.randint(0, 200) for _ in range(3))
    color = (255, 153, 0)
    
    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, WIDTH)
    string = f"ID: {str(pedestrian_id)}"
    location = (x1, y1 - 10)
    cv2.putText(frame, string, location, FONT,SIZE, color, WIDTH)
    
    return frame

def draw_gesture_labels(frame, df_video_labels, frame_id, x1, y1, pedestrian_id):
    """ Draw gesture labels on the frame from the DataFrame. """
    
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SIZE = 0.5
    WIDTH = 1
        
    # Filter for the current pedestrian and frame
    df_video_labels = df_video_labels[
        (df_video_labels["pedestrian_id"] == pedestrian_id) &
        (df_video_labels["start_frame"] <= frame_id) &
        (df_video_labels["end_frame"] >= frame_id)
    ]
    if df_video_labels.empty:
        return frame

    # Get the first relevant label row
    row = df_video_labels.iloc[0]

    # Labels we're interested in (only keep ones present)
    LABEL_KEYS = {"gesture_label_id": "Gesture", "ego_driver_mask": "Ego Driver"}
    label_values = {
        value: row[key]
        for key, value in LABEL_KEYS.items()
        if key in row
    }

    # Draw available labels
    for idx, (label_name, label_value) in enumerate(label_values.items()):
        if pd.isna(label_value):  # skip NaN values
            continue
        
        # Generic label text and color
        label_text = f"{label_name}: {label_value}"
        color = (0, 0, 255) if label_value == 0 else (0, 255, 0)
        
        # Specific label handling
        if label_name == "Gesture":
            label_text += f" ({Gesture(label_value).name})"
            color = (
                Gesture(label_values["Gesture"]).color
                if "Gesture" in label_values
                else (255, 153, 0)
            )

        # Set the position for the label text 
        label_position = (x1, y1 - 10 - (idx+1) * 25)
        cv2.putText(frame, label_text, label_position, FONT, SIZE, color, WIDTH)
    
    return frame

def draw_pedestrians(frame, df_video, frame_id, width, height, df_video_labels):
    """ Draw pedestrians and their bounding boxes on the frame. """
    
    # Get the pedestrian IDs for the current frame
    pedestrian_ids = df_video[df_video["frame_id"] == frame_id]["pedestrian_id"]
    if len(pedestrian_ids) == 0:
        return frame
    
    # Check for duplicate pedestrian IDs
    if pedestrian_ids.duplicated().any():
        location = (width // 2 - 200, height // 2)
        cv2.putText(
            frame, f"Duplicate IDs detected\n{pedestrian_ids}", location,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA
        )
    
    for i, pedestrian_id in enumerate(pedestrian_ids):
        pedestrian = get_pedestrian(df_video, frame_id, pedestrian_id)
        if pedestrian is None:
            continue

        x1, y1, x2, y2 = get_bbox_from_id(pedestrian, width, height)
        draw_bbox_id(frame, x1, y1, x2, y2, pedestrian_id)
        draw_gesture_labels(frame, df_video_labels, frame_id, x1, y1, pedestrian_id)
    
    return frame

def draw_info(frame, video_name, frame_id, interval):
    """ Draw the video name, frame number, and interval on the frame. """
    
    info = [
        f'Video: {video_name}',
        f'Frame: {frame_id}',
        f'Interval: {interval}'
    ]
    for i, text in enumerate(info):
        cv2.putText(
            frame, text, (20, 50 + i * 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA
        )
    
    return frame

def visualize_video(video_path, df, df_video_labels=None):
    global interval
    
    # Check if the HUD should be displayed
    global show_hud
    if not show_hud:
        return frame
    
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
        if show_hud:
            frame = draw_pedestrians(frame, df_video, frame_id, width, height, df_video_labels)
            frame = draw_info(frame, video_name, frame_id, interval)
        
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