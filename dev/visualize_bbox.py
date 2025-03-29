# %%

import cv2
import pandas as pd
import os
import sys
sys.path.append(".")

def visualize_results(csv_path, videos_folder):

    # Check if the video folder and CSV file exist
    if not os.path.exists(videos_folder):
        raise FileNotFoundError(f"Video folder {videos_folder} does not exist.")
    if not os.path.isdir(videos_folder):
        raise NotADirectoryError(f"Video folder {videos_folder} is not a directory.")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist.")
    if not os.path.isfile(csv_path):
        raise NotADirectoryError(f"CSV file {csv_path} is not a file.")

    # Load the CSV file
    df = pd.read_csv(csv_path, index_col=False) if os.path.exists(csv_path) else None
    
    # Get videos by unique video names in the CSV file
    video_names = df["video_name"].unique() if df is not None else []
    if len(video_names) == 0:
        print("Error: No video names found in the CSV file.")
        return
    
    for video_name in video_names:
        video_path = os.path.join(videos_folder, video_name)
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} does not exist.")
            continue
        if not os.path.isfile(video_path):
            print(f"Error: Video file {video_path} is not a file.")
            continue
        if not video_path.endswith(('.mp4', '.avi', '.mov')):
            print(f"Error: Video file {video_path} is not a valid video format.")
            continue
        
        # Visualize the video with bounding boxes
        visualize_video(video_path, df)

def control_video_playback(play, frame_id, total_frames):
    """ Control video playback with keyboard input. """
    
    # Get key press
    key = cv2.waitKey(10) if play else cv2.waitKeyEx(0)
    
    # Control playback state
    play = not play if  key == 32   else play # Space to toggle play/pause
    exit() if           key == 113  else None # 'q' to exit
    
    # Control playback speed and frame navigation
    frame_id += 1 if play           else 0 # Play mode
    frame_id += 1 if key == 2555904 else 0 # Right arrow
    frame_id -= 1 if key == 2424832 else 0 # Left arrow
    
    # Keep frame_id within bounds
    frame_id = max(0, min(frame_id, total_frames - 1))
    
    return play, frame_id

def draw_bounding_boxes(frame, df_video, frame_id, width, height):
    
    pedestrian_ids = df_video[df_video["frame_id"] == frame_id]["pedestrian_id"].unique()
    if len(pedestrian_ids) == 0:
        # print(f"Error: No pedestrian IDs found for frame {frame_id}.")
        return frame
    
    for pedestrian_id in pedestrian_ids:
        # Filter the DataFrame for the current pedestrian ID
        df_pedestrian = df_video[
            (df_video["pedestrian_id"] == pedestrian_id) & 
            (df_video["frame_id"] == frame_id)
        ]
        if df_pedestrian.empty:
            # print(f"Error: No data found for pedestrian {pedestrian_id} in frame {frame_id}.")
            continue

        # Get the bounding box coordinates
        x1_norm, y1_norm, x2_norm, y2_norm = df_pedestrian.iloc[0][["x1", "y1", "x2", "y2"]].values     
        x1, y1, x2, y2 = x1_norm * width, y1_norm * height, x2_norm * width, y2_norm * height
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw the bounding box on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(pedestrian_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def visualize_video(video_path, df):
    
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
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize variables
    play = False
    frame_id = 0

    # Create a window to display the video
    while cap.isOpened():
        
        # Check if the video is playing or paused and read the frame
        if not play: cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        if not ret: break
        
        # Draw bounding boxes on the frame
        frame = draw_bounding_boxes(frame, df_video, frame_id, width, height)
        cv2.putText(
            frame, f'Frame: {frame_id}', (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Processed Video', frame)
        play, frame_id = control_video_playback(play, frame_id, total_frames)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    # Specify the paths to the CSV file and video folder
    csv_path = "data/labels/annotations.csv"
    videos_folder = "data/videos/"

    visualize_results(csv_path, videos_folder)