# %%

import cv2
import pandas as pd
import os

import sys
sys.path.append(".")

def vis_results(video_path: str, csv_path: str, keyword: str):

    # Load the CSV file if it exists
    df = pd.read_csv(csv_path, index_col=False) if csv_path is not None and os.path.exists(csv_path) else None

    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Break the loop if the video is over
        if not ret: break
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
        
        # Display the frame
        frame = cv2.resize(frame, (1280, 720))
        cv2.imshow('Processed Video', frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Visualize results.")
    parser.add_argument("--video_path", type=str, help="Path to the video.", required=True)
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument("--keyword", type=str, default="label", help="Keyword to visualize.")
    args = parser.parse_args()
    
    # Example usage:
    """
    python visulise.py \
        --video_path "data/videos/video_0153.mp4" \
        --csv_path   "data/sanity/output/pred_actions.csv" \
        --keyword    "label"
    """
    
    vis_results(args.video_path, args.csv_path, args.keyword)