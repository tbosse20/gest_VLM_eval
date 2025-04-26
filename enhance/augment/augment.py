import os, sys
import mediapipe as mp
import cv2
from ultralytics import YOLO
import sys
from tqdm import tqdm

sys.path.append(".")
from enhance.body_description.body_description import desc_person, write_desc
import enhance.video_pipeline as video_pipeline

# Suppress ULtralytics logging
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# ——— Models (load once) ———
yolo = YOLO("weights/yolov8n-pose.pt")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_styles = mp.solutions.drawing_styles

# Instantiate MediaPipe Holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,  # video mode
    model_complexity=2,  # highest accuracy
    smooth_landmarks=True,  # temporal smoothing
    refine_face_landmarks=True,  # improved face/hand detail
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def draw_face_landmarks(landmark_list, overlay, roi_coordinates):
    """
    Draw face landmarks on `overlay`, skipping finger landmarks and wrist→hand bones.

    Args:
        - landmark_list: results.pose_landmarks
        - overlay: full‐frame image to draw onto
        - roi_coordinates: (x1,y1,x2,y2) in full‐frame pixel coords
    """
    if not landmark_list:
        return overlay

    x1, y1, x2, y2 = roi_coordinates

    mp_drawing.draw_landmarks(
        overlay[y1:y2, x1:x2],
        landmark_list,
        mp_holistic.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style(),
    )

    return overlay


def draw_pose_landmarks(landmark_list, overlay, roi_coordinates):
    """
    Draw body pose on `overlay`, skipping finger landmarks and wrist→hand bones.

    Args:
        - landmark_list: results.pose_landmarks
        - overlay: full‐frame image to draw onto
        - roi_coordinates: (x1,y1,x2,y2) in full‐frame pixel coords
    """
    if not landmark_list:
        return overlay

    x1, y1, x2, y2 = roi_coordinates

    mp_drawing.draw_landmarks(
        overlay[y1:y2, x1:x2],
        landmark_list,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
    )

    return overlay


def draw_hand_landmarks(landmark_list, overlay, roi_coordinates, left=True):
    """
    Draw hand landmarks on overlay[y1:y2, x1:x2].
    landmark_list must be results.left_hand_landmarks or right_hand_landmarks.
    """
    if not landmark_list:
        return overlay

    x1, y1, x2, y2 = roi_coordinates

    # choose color based on left/right if you like
    color = (0, 0, 255) if left else (255, 0, 0)
    landmark_spec = mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
    connection_spec = mp_drawing.DrawingSpec(color=color, thickness=2)

    mp_drawing.draw_landmarks(
        overlay[y1:y2, x1:x2],
        landmark_list,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=landmark_spec,
        connection_drawing_spec=connection_spec,
    )
    return overlay


def draw_landmarks(res, overlay, roi_coords):

    # draw body & hands
    draw_face_landmarks(res.face_landmarks, overlay, roi_coords)
    draw_pose_landmarks(res.pose_landmarks, overlay, roi_coords)
    draw_hand_landmarks(res.left_hand_landmarks, overlay, roi_coords, left=True)
    draw_hand_landmarks(res.right_hand_landmarks, overlay, roi_coords, left=False)


def get_roi_coords(box, W, H, pad=0.1):
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    pad = int(0.1 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
    roi_coords = (x1, y1, x2, y2)

    return roi_coords


def process_frame(frame, draw: int = 0) -> tuple:
    """
    Given a YOLO result yres and frame, crop each person, run holistic,
    and draw pose (filtered) + both hands onto a semi-transparent overlay.
    """

    if isinstance(frame, str):
        frame = cv2.imread(frame)

    overlay = frame.copy()
    H, W, _ = frame.shape

    yres = yolo(frame)[0]
    
    descriptions = []
    for box in yres.boxes[yres.boxes.cls == 0]:

        roi_coords = get_roi_coords(box, W, H)
        x1, y1, x2, y2 = roi_coords

        # Process the cropped region of interest (ROI) with Holistic
        roi = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        # Draw landmarks on the overlay
        draw_landmarks(res, overlay, roi_coords)

        # Describe the person
        hands_list = [
            (res.left_hand_landmarks, "Left"),
            (res.right_hand_landmarks, "Right"),
        ]
        description = desc_person(res.face_landmarks, hands_list)
        write_desc(overlay, description, pos=(x1, y1)) if draw >= 2 else None

        descriptions.append(description)

    flattened = [item for sublist in descriptions for item in sublist]
    descriptions = " ".join(flattened)

    return overlay, descriptions


if __name__ == "__main__":

    video_pipeline.main(process_frame)
    holistic.close()
