# import object_detect
# import pose
import cv2
import llama2
# import platform
# import vllama2, prompts
import torch
import gc

def print_gpu_memory():
    """Prints the current GPU memory usage."""
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        cached_memory = torch.cuda.memory_reserved() / (1024 * 1024) # Convert to MB
        print(f"GPU Allocated Memory: {allocated_memory:.2f} MB")
        print(f"GPU Reserved Memory: {cached_memory:.2f} MB")
    else:
        print("CUDA is not available.")

def generate_prompt(frame):

    """ Process the input frame """
    print_gpu_memory()
    
    # Pipeline design flags
    PROJECT_POSE    = True
    CAPTION_OBJECTS = True

    # # Pose detection and caption
    # pose_captions = pose.main(frame, project_pose=PROJECT_POSE)
    # pose_captions = None
    # # Implicit object detection, excluding people
    # object_captions = object_detect.main(frame)
    
    # # Implicit sign detection
    # # TODO
    
    # ### Visual Language Model ###
    # # Analyze the frame
    # frame_output = vllama2.inference(frame, prompts.frame, "image")
    # # frame_output = "FAKE FRAME OUTPUT"
    
    # # Concatenate captions into a single string
    # complete_caption = " ".join([frame_output] + pose_captions)
    # print(complete_caption)
    
    init_prompt = """
        You are given a description of a scene from a dash-cam perspective. Your task is to evaluate the situation and suggest the best course of action for the ego driver to take. The description includes details about pedestrians, vehicles, and street conditions. Based on this information, your goal is to choose one of the following actions and provide a clear explanation of why that action is the most appropriate.

        The scene details are as follows:
    """

    complete_caption_example = """
        The image depicts a narrow street with parked cars on both sides. There are at least six cars visible, including a white van and a black car in the foreground. A man is walking down the street, carrying a child, who is wearing a blue shirt. The scene suggests that the vehicle is navigating through this busy street, possibly looking for a parking spot or driving to a destination. The presence of pedestrians and parked cars indicates that the driver needs to be cautious and attentive to avoid any accidents. 0. The pedestrian in the image is standing with their arms outstretched, possibly signaling a need for assistance or indicating a specific direction. Their body language suggests that they are trying to communicate with a vehicle, perhaps requesting it to stop or slow down. The presence of lines connecting different parts of their body further emphasizes the importance of their gestures and adds a visual element to their communication. 1. The pedestrian in the image is walking towards a vehicle, with their arms outstretched and legs slightly bent. Their body posture suggests that they are signaling to the driver of the vehicle to stop or slow down. The pedestrian's outstretched arms and forward-leaning stance indicate a clear intention to communicate with the driver. Based on these cues, it can be inferred that the pedestrian is attempting to get the driver's attention and convey a message related to traffic safety.
    """
    complete_caption = complete_caption_example

    task_prompt = """
        ### Possible Actions:
        1. Accelerate
        2. Decelerate
        3. Brake
        4. Turn left
        5. Turn right

        ### Instructions:
        - **Choose only one action.**
        - **Provide only one response** in the form of a JSON object with two keys: `"action"` and `"reason"`.
        - `"action"`: The selected action (one of the possible actions above).
        - `"reason"`: A short explanation for why this action is the most appropriate.

        The driver should: 
    """

    # Interpret the complete caption
    complete_prompt = "\n".join([init_prompt, complete_caption, task_prompt])

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print_gpu_memory()

    return complete_prompt

def decide_action(complete_prompt):
    action = llama2.inference(complete_prompt)
    return action

def caption_frame(frame):
    complete_prompt = generate_prompt(frame)
    action = decide_action(complete_prompt)
    return action

if __name__ == "__main__":
    image_path = 'data/sanity/video_0153.png'
    frame = cv2.imread(image_path)
    complete_caption = caption_frame(frame)
    print("Complete caption:", complete_caption)
