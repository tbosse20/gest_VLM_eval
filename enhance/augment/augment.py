import os, sys

# 1) Tell glog (used by MediaPipe C++) to only show FATAL messages
os.environ["GLOG_minloglevel"] = "3"  # 0=INFO,1=WARNING,2=ERROR,3=FATAL only
os.environ["GLOG_logtostderr"] = "1"  # redirect to stderr so minloglevel applies

# 2) (Optional) If you also use TF, silence its logs below ERROR
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 3) Silence Abseil‐side Python logs (MediaPipe wrapper uses absl)
from absl import logging as _absl_logging

_absl_logging.set_verbosity(_absl_logging.FATAL)
_absl_logging.set_stderrthreshold(_absl_logging.FATAL)

# 4) Temporarily divert stderr to null while we import MediaPipe
_old_stderr = sys.stderr
sys.stderr = open(os.devnull, "w")

# 5) Now import the native‐backed modules
import mediapipe as mp
import cv2
from ultralytics import YOLO

# …any other native imports…

# 6) Restore stderr so your own prints still show
sys.stderr.close()
sys.stderr = _old_stderr

from tqdm import tqdm

# at top‐level, once:
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Suppress ULtralytics logging
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("mediapipe").setLevel(logging.ERROR)


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
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
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
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
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


def draw_landmarks(frame):
    """
    Given a YOLO result yres and frame, crop each person, run holistic,
    and draw pose (filtered) + both hands onto a semi-transparent overlay.
    """

    overlay = frame.copy()
    H, W, _ = frame.shape

    yres = yolo(frame)[0]

    for box in yres.boxes[yres.boxes.cls == 0]:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        pad = int(0.1 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(W, x2 + pad), min(H, y2 + pad)
        roi_coords = (x1, y1, x2, y2)

        roi = frame[y1:y2, x1:x2]
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = holistic.process(rgb)

        # draw body & hands
        draw_face_landmarks(res.face_landmarks, overlay, roi_coords)
        draw_pose_landmarks(res.pose_landmarks, overlay, roi_coords)
        draw_hand_landmarks(res.left_hand_landmarks, overlay, roi_coords, left=True)
        draw_hand_landmarks(res.right_hand_landmarks, overlay, roi_coords, left=False)

    return overlay


# ——— Models (load once) ———
yolo = YOLO("weights/yolov8n-pose.pt")
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
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


def pose_from_video(video_path: str, video_output: str):

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"{video_path} not found")
    if not video_path.endswith((".mp4", ".avi", ".mov", ".MP4")):
        raise ValueError(f"{video_path} is not a video file")

    # Check if the video file is a valid video file
    os.makedirs(os.path.dirname(video_output), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_output, fourcc, fps, (W, H))

    video_name = os.path.basename(video_path)
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))), desc=video_name):
        ret, frame = cap.read()
        if not ret:
            break

        frame = draw_landmarks(frame)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Done → {video_output}")


def pose_dir(videos_dir: str, extension: str = "augmented"):

    # Check if the main folder exists
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"{videos_dir} not found")

    # Make sibling folder to videos_dir
    parent_dir = os.path.dirname(videos_dir)
    output_dir = os.path.join(parent_dir, extension)
    os.makedirs(output_dir, exist_ok=True)

    for video in tqdm(os.listdir(videos_dir), desc="Processing"):

        # Get the full path to the video file and output path
        video_path = os.path.join(videos_dir, video)
        video_output = os.path.join(output_dir, video)

        # Check if the video file exists and is a valid video file
        if not os.path.isfile(video_path):
            print(f"{video_path} not found")
            continue
        if not video_path.endswith((".mp4", ".avi", ".mov", ".MP4")):
            print(f"{video_path} is not a video file")
            continue

        # Process the video
        pose_from_video(video_path, video_output)
        break


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Pose estimation from video")
    parser.add_argument(
        "input", type=str, help="Path to the input video folder or video file."
    )
    args = parser.parse_args()

    # Example usage:
    """ 
        python enhance/augment.py /path/to/videos_dir
    """

    if os.path.isdir(args.input):
        pose_dir(args.input)
    elif os.path.isfile(args.input):
        output_dir = os.path.join(os.path.dirname(args.input), "augmented")
        pose_from_video(args.input, output_dir)
    else:
        raise FileNotFoundError(f"{args.input} not found")
    holistic.close()
