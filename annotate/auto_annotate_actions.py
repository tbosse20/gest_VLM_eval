
import pandas as pd
import os
import action

def generate_action_csv(action_csv: str):
    """ Generate an empty action CSV file """
    
    # Remove existing action CSV
    if os.path.exists(action_csv): 
        os.remove(action_csv)
    
    # Create an empty CSV file
    df = pd.DataFrame(columns=["video_name", "frame_idx", "action", "action_id"])
    df.to_csv(action_csv, mode="w", index=False, header=True)

def annotate_folder(folder_path: str, action_csv: str): 
    """ Annotate all videos in a folder """
    
    # Generate or replace action CSV
    generate_action_csv(action_csv)
    
    # Get all video files in the folder
    video_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
    
    # Annotate each video
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        annotate_video(video_path,action_csv)
        
def annotate_video(video_path: str, action_csv: str):
    
    # Generate or replace action CSV
    generate_action_csv(action_csv)
    
    # Extract actions from video
    df = action.extract_action_video(video_path)
    
    # Save to CSV
    df.to_csv(action_csv, mode="a", index=False, header=False)

if __name__ == "__main__":
    folder_path = 'data/sanity'
    action_csv = 'data/sanity/actions.csv'
    # annotate_folder(folder_path, action_csv)
    annotate_video('data/sanity/video_0153.mp4', action_csv)