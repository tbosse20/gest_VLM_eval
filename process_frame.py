import object_detect, pose
import cv2
import platform
import vllama2, prompts

def caption_frame(frame):
    """ Process the input frame """
    
    # Pipeline design flags
    PROJECT_POSE    = True
    CAPTION_OBJECTS = True

    # Pose detection and caption
    pose_captions = pose.main(frame, project_pose=PROJECT_POSE)
    # Implicit object detection, excluding people
    object_captions = object_detect.main(frame)
    
    # Implicit sign detection
    # TODO
    
    ### Visual Language Model ###
    # Analyze the frame
    frame_output = vllama2.inference(frame, prompts.frame, "image")
    # frame_output = "FAKE FRAME OUTPUT"
    
    # Concatenate captions into a single string
    complete_caption = ".\n".join([frame_output] + pose_captions + object_captions)
    
    # Interpret the complete caption
    # TODO
    
    return complete_caption

if __name__ == "__main__":
    image_path = 'data/sanity/video_0153.png'
    frame = cv2.imread(image_path)
    complete_caption = caption_frame(frame)
    print("Complete caption:", complete_caption)