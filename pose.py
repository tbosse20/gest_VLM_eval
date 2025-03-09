
import torch
from ultralytics import YOLO
import cv2
import logging
import vllama2, prompts
import platform

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
# print(device, dtype, "GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))

# Load YOLOv8 model for person detection
yolo_pose = YOLO("weights/yolov8n-pose.pt", verbose=False)  # Small model
yolo_pose.to(device).eval()

# Define pedestrian detection functions
def inference(frame):
    """ Draw bounding boxes around detected pedestrians """

    with torch.no_grad():
        results = yolo_pose(frame)
        
    # Extract the first result
    result = results[0]
    
    return result

def get_crop(frame, box):
    """ Get the cropped region of interest """
    x1, y1, x2, y2 = map(int, box[:4])
    return frame[y1:y2, x1:x2]

def get_crops(frame, results):
    """ Get the cropped regions of interest """
    crops = []

    if results is None or results.boxes is None:
        return crops

    boxes = results.boxes.data

    # Handle CUDA tensors if present
    if hasattr(boxes, 'is_cuda') and boxes.is_cuda:
        boxes = boxes.cpu().numpy()
    else:
        boxes = boxes.numpy()

    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = map(int, box[:6])

        crop = frame[y1:y2, x1:x2]
        crops.append(crop)

    return crops

def caption_crops(pose_crops):
    # Analyze each cropped pedestrian w/wo pose
    return [
        f"{i}. {vllama2.inference(pose_crop, prompts.pose, 'image')}"
        # f"{i}. FAKE POSE CROP OUTPUT"
        for i, pose_crop in enumerate(pose_crops)
    ]

def main(frame, project_pose=True):

    # Detect pedestrians
    pose_result = inference(frame)
    # Plot the detected poses
    frame = pose_result.plot() if project_pose else frame
    # Get the cropped region of interest
    pose_crops = get_crops(frame, pose_result)
    # Caption each pedestrian
    caption = caption_crops(pose_crops)

    return caption

if __name__ == "__main__":
    
    # Load an image
    frame = cv2.imread("data/sanity/video_0153.png")
    
    # Detect pedestrians
    captions = main(frame, project_pose=False)

    # Print captions
    for i, caption in enumerate(captions):
        print(f'{caption}')

    # Display the result
    if platform.system() == "Linux": exit()
    cv2.imshow("Pedestrian Pose", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()