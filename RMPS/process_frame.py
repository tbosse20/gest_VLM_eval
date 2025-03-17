
import cv2
import torch
import gc

import sys
sys.path.append(".")
import RMPS.caption_video as caption_video
import scripts.decide_action as decide_action

def caption_video(frame):
    complete_prompt = caption_video.generate_prompt(frame)
    action = decide_action.decide_action(complete_prompt)
    return action

if __name__ == "__main__":
    image_path = 'data/sanity/input/video_0153.png'
    frame = cv2.imread(image_path)
    complete_caption = caption_video(frame)
    print("Complete caption:", complete_caption)
