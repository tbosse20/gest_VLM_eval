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

def inference(frame):
    """ Detect objects in the input frame """
    
    with torch.no_grad():
        results = yolo(frame)
        
    # Extract the first result
    result = results[0]
    
    return result

def get_crop(frame, box):
    """ Get the cropped region of interest """
    x1, y1, x2, y2 = map(int, box[:4])
    return frame[y1:y2, x1:x2]

def get_crops(frame, results, exclude_classes=[]):
    """ Get the cropped region of interests """
    boxes = (
        results.boxes.data.cpu().numpy()
        if results.boxes.data.is_cuda
        else results.boxes.data.numpy()
    )
    crops = [
        get_crop(frame, box)
        for box in boxes
        if int(box[5]) not in exclude_classes
    ]
    
    return crops

if __name__ == "__main__":
    
    # Load an image
    frame = cv2.imread("data/sanity/video_0153.png")
    
    # Detect objects
    result = inference(frame)
    
    # Plot the detected objects
    frame = result.plot()
    
    # Display the result
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows