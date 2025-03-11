
import cv2
import torch
import gc
import caption_frame
import decide_action

def caption_frame(frame):
    complete_prompt = caption_frame.generate_prompt(frame)
    action = decide_action.decide_action(complete_prompt)
    return action

if __name__ == "__main__":
    image_path = 'data/sanity/video_0153.png'
    frame = cv2.imread(image_path)
    complete_caption = caption_frame(frame)
    print("Complete caption:", complete_caption)
