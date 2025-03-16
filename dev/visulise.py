import cv2
import pandas as pd
import os

import sys
sys.path.append(".")
import dev.dev_utils as dev_utils

def vis_results(video_path, csv_path, keyword: str):

    df = pd.read_csv(csv_path, index_col=False) if os.path.exists(csv_path) else None

    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object
    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # out = cv2.VideoWriter('data/sanity/output/output.mp4', fourcc, fps, (width, height))
    
    prev_time = 0  # For fps calculation
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Break the loop if the video is over
        if not ret: break

        # Add current frame rate to the frame shown
        # frame, prev_time = dev_utils.display_fps(frame, prev_time)
        frame_counter += 1

        cv2.putText(
            frame, f'Frame: {frame_counter}', (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Display FPS on the frame
        if df is not None:
            action = df[df["frame_idx"] == frame_counter][keyword].values[0]
            cv2.putText(
                frame, f'Pred. Action: {action}', (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # # Write the frame to the output video
        # out.write(frame)

        # Display the frame
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    video_path = "data/videos/video_0153.mp4"
    # csv_path = "data/sanity/output/pred_actions.csv"
    # csv_path = "data/labels/video_0153.csvs"
    csv_path = ""

    vis_results(video_path, csv_path, keyword='label')