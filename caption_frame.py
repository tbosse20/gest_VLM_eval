import cv2
import object_detect
import pose
import torch
import gc
import os
import pandas as pd
import re
import vllama2, prompts

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

    # complete_caption_example = """
    #     The image depicts a narrow street with parked cars on both sides. There are at least six cars visible, including a white van and a black car in the foreground. A man is walking down the street, carrying a child, who is wearing a blue shirt. The scene suggests that the vehicle is navigating through this busy street, possibly looking for a parking spot or driving to a destination. The presence of pedestrians and parked cars indicates that the driver needs to be cautious and attentive to avoid any accidents. 0. The pedestrian in the image is standing with their arms outstretched, possibly signaling a need for assistance or indicating a specific direction. Their body language suggests that they are trying to communicate with a vehicle, perhaps requesting it to stop or slow down. The presence of lines connecting different parts of their body further emphasizes the importance of their gestures and adds a visual element to their communication. 1. The pedestrian in the image is walking towards a vehicle, with their arms outstretched and legs slightly bent. Their body posture suggests that they are signaling to the driver of the vehicle to stop or slow down. The pedestrian's outstretched arms and forward-leaning stance indicate a clear intention to communicate with the driver. Based on these cues, it can be inferred that the pedestrian is attempting to get the driver's attention and convey a message related to traffic safety.
    # """
    # complete_caption = complete_caption_example

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return complete_caption

def caption_video(video_path, csv_path, vllama2_package=None, sanity=False):
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

    vllama2_package = vllama2_package or vllama2.load_model()

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Break the loop if the video is over
        if not ret: break

        # Process the frame
        complete_caption = generate_prompt(frame, vllama2_package)
        # print("Complete caption:", complete_caption)

        # Write to csv
        video_name = os.path.basename(video_path)
        re_complete_caption = re.sub(r'\s+', ' ', complete_caption).strip()
        df = pd.DataFrame({"video_name": [video_name], "frame_idx": [frame_idx], "caption": [re_complete_caption]}, )
        df.to_csv(csv_path, mode="a", index=False, header=False)

        ### Finalize ### 
        # Add current frame rate to the frame shown
        frame_idx += 1

        # Sanity check
        if frame_idx > 5 and sanity: break
    
    # Release the VideoCapture and VideoWriter objects
    cap.release()
    cv2.destroyAllWindows()

    gc.collect()
    torch.cuda.empty_cache()

def caption_folder(folder_path):
    pass

if __name__ == "__main__":
    video_path = "data/sanity/video_0153.mp4"
    csv_path = "data/sanity/caption.csv"
    caption_video(video_path, csv_path, sanity=True)