
import torch
from ultralytics import YOLO
import cv2
import logging

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# print(device, dtype, "GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))

# Load YOLOv8 model for person detection
yolo_pose = YOLO("weights/yolov8n-pose.pt", verbose=False)  # Small model
yolo_pose.to(device).eval()

# Define pedestrian detection functions
def detect_pedestrians(frame):
    """ Draw bounding boxes around detected pedestrians """

    with torch.no_grad():
        results = yolo_pose(frame)
        
    # Extract the first result
    result = results[0]
    
    return result

def plot_pose(result):
    """ Plot the detected poses on the input frame """
    return result.plot()

if __name__ == "__main__":
    
    # Load an image
    frame = cv2.imread("data/sanity/video_0153.png")
    
    # Detect pedestrians
    result = detect_pedestrians(frame)
    
    # Plot the detected poses
    frame = plot_pose(result)
    
    # Display the result
    cv2.imshow("Pedestrian Pose", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()