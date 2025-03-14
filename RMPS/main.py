import cv2
import os
import platform

import dev.dev_utils as dev_utils
from scripts.process_frame import caption_frame

def process_folder(folder_path):
    """ Process the images in the input folder """
    video_paths = os.listdir(folder_path)
    
    for video_path in video_paths:
        process_video(video_path)

def process_video(video_path):
    """ Process the input video """
    
    # Load the video
    cap = cv2.VideoCapture(video_path)
    
    # Get the video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter('data/sanity/output/output.mp4', fourcc, fps, (width, height))
    
    prev_time = 0  # For fps calculation
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        # Break the loop if the video is over
        if not ret: break

        # Process the frame
        complete_caption = caption_frame(frame)
        print("Complete caption:", complete_caption)
        
        # Add current frame rate to the frame shown
        frame, prev_time = dev_utils.display_fps(frame, prev_time)
        frame_counter += 1
                
        # Write the frame to the output video
        out.write(frame)

        # Display the frame
        if platform.system() != "Linux":
            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('Processed Video', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if frame_counter > 5: break
    
    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = 'data/sanity/input/video_0153.mp4'
    process_video(video_path)