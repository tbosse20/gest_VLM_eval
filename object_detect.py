from ultralytics import YOLO
import torch
import cv2
import logging
import platform
import vllama2, prompts

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

def get_crops(frame, results, exclude_pedestrians):
    """ Get the cropped region of interests """
    boxes = (
        results.boxes.data.cpu().numpy()
        if results.boxes.data.is_cuda
        else results.boxes.data.numpy()
    )
    crops = [
        get_crop(frame, box)
        for box in boxes
        if int(box[5]) != 0 and exclude_pedestrians
    ]
    
    return crops


def caption_crops(object_crops):
    # Analyze each cropped object
    return [
        f"{i}. {vllama2.inference(object_crop, prompts.object, 'image')}"
        # f"{i}. FAKE OBJECT CROP OUTPUT"
        for i, object_crop in enumerate(object_crops)
    ]

def main(frame, exclude_pedestrians=True):

    # Detect objects in the frame
    object_result = inference(frame)
    # Plot the detected objects
    frame = object_result.plot()
    # Get the cropped region of interest
    object_crops = get_crops(frame, object_result, exclude_pedestrians)
    # Caption each object
    caption = caption_crops(object_crops)

    return caption

if __name__ == "__main__":
    
    # Load an image
    frame = cv2.imread("data/sanity/video_0153.png")
    
    # Detect pedestrians
    captions = main(frame, exclude_pedestrians=True)

    # Print captions
    for i, caption in enumerate(captions):
        print(f'{caption}')

    # Display the result
    if platform.system() == "Linux": exit()
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow("Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows