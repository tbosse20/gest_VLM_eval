import cv2
import sys, os
sys.path.append("../../VideoLLaMA2")
import gc
import torch
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

action_categories = ["Constant Speed", "Accelerating", "Decelerating", "Turning Left", "Turning Right", "Halted"]

# from videollama2 import model_init, mm_infer
# from videollama2.utils import disable_torch_init
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()
# disable_torch_init()

# Load the VideoLLaMA2 model
# model_path = 'DAMO-NLP-SG/VideoLLaMA2.1-7B-16F'
# model, processor, tokenizer = model_init(model_path)

# # Function to create a video from images
# def create_video(frames, output_video_path, fps):

#     # Get frame size
#     height, width, layers = frames[0].shape
#     size = (width, height)

#     # Create video writer
#     out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps=1, size=size)

#     for img in frames:
#         out.write(img)

#     out.release()

# def classify_driver_action(frames) -> str:
#     """
#     Analyzes three consecutive frames to understand the ego driver's actions.

#     Args:
#         frames: ...
#     """

#     OUTPUT_PATH = "tmp_output_video.mp4"
#     MODAL = "video"

#     # MAKE ERROR
#     if len(frames) < 3 or 3 < len(frames):
#         return None
    
#     create_video(frames, OUTPUT_PATH, fps=1)

#     # Analyze Driver's Current State (Focus on the last frame)
#     instruct_driver_state = """
#     Examine the following three consecutive video frames. Focus specifically on the ego vehicle's state in the *last* (third) frame. Based on the changes observed across all three frames, but with an emphasis on the third frame, select the most accurate description of the driver's *current* driving state:

#     1. Constant Speed: The vehicle is maintaining a steady pace.
#     2. Accelerating:   The vehicle is actively increasing its speed.
#     3. Decelerating:   The vehicle is actively decreasing its speed.
#     4. Turning Left:   The vehicle is in the process of turning left.
#     5. Turning Right:  The vehicle is in the process of turning right.
#     6. Halted:         The vehicle is completely stopped.

#     Provide your answer as the number and title corresponding to the best description of the driver's *current* state in the last frame. For example, answer '2. Accelerating' if the driver is accelerating in the last frame.
#     """
    
#     output = mm_infer(processor[MODAL](OUTPUT_PATH), instruct_driver_state, model=model, tokenizer=tokenizer, do_sample=False, modal=MODAL)

#     return output

def extract_action_video(video_path, save=False, sanity=False):

    # Run video and action recognition
    cap = cv2.VideoCapture(video_path)
    prev_frames = []
    frame_counter = 0

    # Define the codec and create a VideoWriter object
    size = (1920, 1080)
    if save:
        out = cv2.VideoWriter("action_capture.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 25, size)
    
    # Initialize lists to store the results
    actions, frame_idx = [], []
    
    # Loop through the video
    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret: 
            logging.info("End of video reached.")
            break
        
        # Append the frame to the list of previous frames
        prev_frames.append(frame.copy())
        
        # Keep only the last 3 frames
        if len(prev_frames) > 3:
            prev_frames.pop(0)
            
        # Classify the driver's action
        if len(prev_frames) == 3:
            # result = classify_driver_action(prev_frames)
            result = "Constant Speed" # Temporary
            
            # Append the result to the list
            actions.append(result)
            frame_idx.append(frame_counter)
            
            # Print the result
            # print(f"Frame {frame_counter}: {result}")
            cv2.putText(frame, f'Action: {result}', (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Display the frame
        frame_counter += 1
        
        # Save the frame to the output video
        if save:
            frame = cv2.resize(frame, size)
            out.write(frame)
        
        # Break if in sanity mode
        if sanity and frame_counter > 10:
            break

    
    if save:
        out.release()
    gc.collect()
    cap.release()
    
    # Make pandas dataframe with frame_idx and action
    video_name = os.path.basename(video_path)
    action_ids = [action_categories.index(action) for action in actions]
    df = pd.DataFrame({"video_name": video_name, "frame_idx": frame_idx, "action": actions, "action_id": action_ids})
    
    return df
    

if __name__ == "__main__":
    video_path = 'data/sanity/video_0153.mp4'
    df = extract_action_video(video_path, save=True, sanity=True)
    print(df)