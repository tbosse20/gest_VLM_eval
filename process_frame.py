import object_detect, pose
import cv2
import platform
if platform.system() == "Linux":
    import vlm, prompts

def caption_frame(frame, prev_frame):
    """ Process the input frame """
    
    # Pipeline design flags
    PROJECT_POSE    = True
    CAPTION_OBJECTS = True
    
    # Save the original frame
    original_frame = frame.copy()
    
    # Action recognition
    # pass
    
    # Detect pedestrians
    pose_result = pose.inference(frame)
    # Plot the detected poses
    frame = pose_result.plot()
    # Get the cropped region of interest
    pose_crops = pose.get_crops(frame.copy() if PROJECT_POSE else original_frame, pose_result)
    
    # Implicit object detection, excluding people
    object_result = object_detect.inference(frame)
    # Plot the detected objects
    frame = object_result.plot()
    # Get the cropped region of interest
    object_crops = object_detect.get_crops(original_frame, object_result, exclude_classes=[0])
    
    # Implicit sign detection
    # TODO
        
    # Estimate speed
    # pass
    
    ### Visual Language Model ###
    # Analyze the frame
    frame_output = vlm.inference(frame, prompts.frame, vlm.Modal.IMAGE)
    
    # Analyze each cropped pedestrian w/wo pose
    pose_outputs = [
        # f"{i}. {vlm.inference(pose_crop, prompts.pose, vlm.Modal.IMAGE)}"
        f"{i}. FAKE POSE CROP OUTPUT"
        for i, pose_crop in enumerate(pose_outputs)
    ]
    
    # Analyze each cropped object
    object_outputs = [
        # f"{i}. {vlm.inference(object_crop, prompts.object, vlm.Modal.IMAGE)}"
        f"{i}. FAKE OBJECT CROP OUTPUT"
        for i, object_crop in enumerate(object_outputs)
    ]
    
    # Concatenate captions into a single string
    complete_caption = ". ".join([frame_output] + pose_outputs + object_outputs)
    print("Complete caption:", complete_caption)
    
    # Interpret the complete caption
    # TODO
    
    # Return the predicted action
    # TODO
    
    return "FAKE ACTION"

if __name__ == "__main__":
    image_path = 'data/sanity/video_0153.png'
    frame = cv2.imread(image_path)
    caption_frame(frame, None)