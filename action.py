import cv2
import sys, os
sys.path.append("../VideoLLaMA2")
from videollama2 import model_init, mm_infer
from videollama2.utils import disable_torch_init
import gc
import torch
import numpy as np
from moviepy.editor import ImageSequenceClip
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
disable_torch_init()

# Load the VideoLLaMA2 model
model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
model, processor, tokenizer = model_init(model_path)

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

def classify_driver_action(frames) -> str:
    """
    Analyzes three consecutive frames to understand the ego driver's actions.

    Args:
        frames: ...
    """

    OUTPUT_PATH = "output_video.mp4"
    MODAL = "video"

    # MAKE ERROR
    if len(frames) < 3 or 3 < len(frames):
        return None
    
    create_video(frames, OUTPUT_PATH, fps=1)

    # Analyze Driver's Current State (Focus on the last frame)
    instruct_driver_state = """
    Examine the following three consecutive video frames. Focus specifically on the ego vehicle's state in the *last* (third) frame. Based on the changes observed across all three frames, but with an emphasis on the third frame, select the most accurate description of the driver's *current* driving state:

    1. Constant Speed: The vehicle is maintaining a steady pace.
    2. Accelerating:   The vehicle is actively increasing its speed.
    3. Decelerating:   The vehicle is actively decreasing its speed.
    4. Turning Left:   The vehicle is in the process of turning left.
    5. Turning Right:  The vehicle is in the process of turning right.
    6. Halted:         The vehicle is completely stopped.

    Provide your answer as the number and title corresponding to the best description of the driver's *current* state in the last frame. For example, answer '2. Accelerating' if the driver is accelerating in the last frame.
    """
    
    output = mm_infer(processor[MODAL](OUTPUT_PATH), instruct_driver_state, model=model, tokenizer=tokenizer, do_sample=False, modal=MODAL)

    return output    

if __name__ == "__main__":
    # Run video and action recognition
    video_path = 'data/sanity/video_0153.mp4'
    cap = cv2.VideoCapture(video_path)
    prev_frames = []
    frame_counter = 0

    size = (1920, 1080)
    out = cv2.VideoWriter("action_capture.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            logging.info("End of video reached.")
            break
        
        prev_frames.append(frame.copy())
        if len(prev_frames) > 3:
            prev_frames.pop(0)
        if len(prev_frames) == 3:
            result = classify_driver_action(prev_frames)
            
            print(f"Frame {frame_counter}: {result}")
            try:
                cv2.putText(frame, f'Action: {result}', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            except:
                pass
            
        frame_counter += 1
        frame = cv2.resize(frame, size)
        out.write(frame)

    out.release()
    gc.collect()
    cap.release()
    
