from ultralytics import YOLO
import torch
import cv2
import logging

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# print(device, dtype, "GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))

# --- Load YOLOv8 Model ---
yolo = YOLO("weights/yolov8n.pt")
yolo.to(device).eval()

def detect_objects(frame):
    """ Detect objects in the input frame """
    
    with torch.no_grad():
        results = yolo(frame)
        
    # Extract the first result
    result = results[0]
    
    return result

def plot_objects(result):
    """ Plot the detected objects on the input frame """
    return result.plot()

if __name__ == "__main__":
    
    # Load an image
    frame = cv2.imread("data/sanity/video_0153.png")
    
    # Detect objects
    result = detect_objects(frame)
    
    # Plot the detected objects
    frame = plot_objects(result)
    
    # Display the result
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows