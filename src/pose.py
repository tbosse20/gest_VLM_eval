
import torch
from ultralytics import YOLO
import cv2
import logging
import platform

import sys
sys.path.append(".")
# import src.vllama2 as vllama2

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    # print(device, dtype, "GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))

    # Load YOLOv8 model for person detection
    yolo_pose = YOLO("weights/yolov8n-pose.pt", verbose=False)  # Small model
    yolo_pose.to(device).eval()

    return yolo_pose

def unload_model(yolo_pose):
    del yolo_pose
    yolo_pose = None

# Define pedestrian detection functions
def inference(frame, yolo_pose=None):
    """ Draw bounding boxes around detected pedestrians """

    yolo_pose = load_model() or yolo_pose

    with torch.no_grad():
        results = yolo_pose(frame)
        
    # Extract the first result
    result = results[0]

    unload_model(yolo_pose)
    
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

    for i, box in enumerate(boxes):
        x1, y1, x2, y2, confidence, class_id = map(int, box[:6])

        crop = frame[y1:y2, x1:x2]
        crops.append(crop)

        cv2.imwrite(f'saved_image{i}_lines.jpg', crop)

    return crops

# def caption_crops(pose_crops, vllama2_package=None):
#     # Analyze each cropped pedestrian w/wo pose
#     vllama2_package = vllama2_package or vllama2.load_model()
#     return [
#         f"Pedestrian {i}. {vllama2.inference(pose_crop, 'What is this pedestrian gesturing?', 'image', vllama2_package)}"
#         # f"{i}. FAKE POSE CROP OUTPUT"
#         for i, pose_crop in enumerate(pose_crops)
#     ]

def main(frame, project_pose=True, vllama2_package=None):

    # Detect pedestrians
    pose_result = inference(frame)
    # Plot the detected poses
    frame = pose_result.plot() if project_pose else frame
    # Get the cropped region of interest
    pose_crops = get_crops(frame, pose_result)
    # Caption each pedestrian
    # caption = caption_crops(pose_crops, vllama2_package)

    # return caption

if __name__ == "__main__":
    
    # Load an image
    frame = cv2.imread("data/sanity/input/video_0153.png")
    
    # Detect pedestrians
    captions = main(frame, project_pose=False)

    # Print captions
    for i, caption in enumerate(captions):
        print(f'{caption}')

    # Display the result
    # cv2.imshow("Pedestrian Pose", frame)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     exit()