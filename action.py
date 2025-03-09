import cv2
import sys, os
sys.path.append("../VideoLLaMA2")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
from PIL import Image

disable_torch_init()

# Load the VideoLLaMA2 model
model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
model, processor, tokenizer = model_init(model_path)

import cv2
import numpy as np
import torch
from moviepy.editor import ImageSequenceClip

# Function to create a video from images
def create_video(frames, output_video_path = "output_video.mp4", fps=1):

    # Get frame size
    height, width, layers = frames[0].shape
    size = (width, height)

    # Create video writer
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for img in frames:
        out.write(img)

    out.release()
    print(f"Video saved at {output_video_path}")

def classify_driver_action(frames):
    """
    Analyzes three consecutive frames to understand the ego driver's actions.

    Args:
        image_paths: A list of three image file paths representing consecutive frames.
    """

    OUTPUT_PATH = "output_video.mp4"

    # MAKE ERROR
    if len(frames) < 3 or 3 < len(frames):
        return None
    
    create_video(frames, OUTPUT_PATH, fps=1)

    # Analyze Driver Actions
    instruct_driver_action = """
        Based on these three consecutive frames, describe the ego
        driver's actions. Is the driver accelerating, decelerating,
        turning, or maintaining a constant speed and direction? Is
        the driver changing lanes, or is the driver doing something else?
    """
    
    output = mm_infer(processor["video"](OUTPUT_PATH), instruct_driver_action, model=model, tokenizer=tokenizer, do_sample=False, modal=modal)
    print("Driver Action Analysis:\n", output)

if __name__ == "__main__":
    # Run video and action recognition
    video_path = 'data/sanity/video_0153.mp4'
    cap = cv2.VideoCapture(video_path)
    prev_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        prev_frames.append(frame.copy())
        action = classify_driver_action(prev_frames)
        if len(prev_frames) > 3: prev_frames.pop()
    cap.release()
    
