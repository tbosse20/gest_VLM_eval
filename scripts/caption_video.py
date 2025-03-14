import cv2
import torch
import gc
import os
import pandas as pd
import re
from tqdm import tqdm

import sys
sys.path.append(".")
import RMPS.prompts as prompts
import src.object_detect as object_detect
import src.pose as pose
import src.vllama2 as vllama2

def generate_prompt(frame, vllama2_package=None):
    """ Process the input frame """

    vllama2_package = vllama2_package or vllama2.load_model()
    
    # Pipeline design flags
    PROJECT_POSE    = True
    CAPTION_OBJECTS = True

    # Pose detection and caption
    pose_captions = pose.main(frame, PROJECT_POSE, vllama2_package)
    
    # Implicit object detection, excluding people
    # object_captions = object_detect.main(frame)
    
    # Implicit sign detection
    # TODO
    
    ### Visual Language Model ###
    # Analyze the frame
    frame_output = vllama2.inference(frame, prompts.frame, "image", vllama2_package)
    # frame_output = "FAKE FRAME OUTPUT"
    
    # Concatenate captions into a single string
    complete_caption = " ".join([frame_output] + pose_captions)

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return complete_caption

def caption_video(video_path, csv_path, interval=1, vllama2_package=None, sanity=False):
    """ Process the input video """

    gc.collect()
    torch.cuda.empty_cache()

    columns = ["video_name", "frame_idx", "caption"]

    # Generate csv file if not exists
    if not os.path.exists(csv_path):
        df = pd.DataFrame(columns=columns)
        df.to_csv(csv_path, mode="w", index=False, header=True)
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vllama2_package = vllama2_package or vllama2.load_model()

    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Break the loop if the video is over
            if not ret: break

            if frame_idx % interval != 0:
                frame_idx += 1
                pbar.update(1)
                continue

            # Process the frame
            complete_caption = generate_prompt(frame, vllama2_package)
            # print("Complete caption:", complete_caption)

            # Write to csv
            video_name = os.path.basename(video_path)
            re_complete_caption = re.sub(r'\s+', ' ', complete_caption).strip()
            df = pd.DataFrame({"video_name": [video_name], "frame_idx": [frame_idx], "caption": [re_complete_caption]})
            df.to_csv(csv_path, mode="a", index=False, header=False)

            ### Finalize ### 
            # Add current frame rate to the frame shown
            frame_idx += 1
            pbar.update(1)

            # Sanity check
            if frame_idx > 15 and sanity: break
    
    # Release the VideoCapture and VideoWriter objects
    cap.release()
    cv2.destroyAllWindows()

    gc.collect()
    torch.cuda.empty_cache()

def caption_folder(folder_path):
    pass

if __name__ == "__main__":

    video_path = "data/sanity/input/video_0153.mp4"
    csv_path = "data/sanity/output/caption.csv"
    interval = 10

    caption_video(video_path, csv_path, interval, sanity=False)
    # caption_video(video_path, csv_path, interval, sanity=True)