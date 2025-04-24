#!/usr/bin/env python3
import os
import cv2
import mediapipe as mp
from tqdm import tqdm

# ——— Silence MediapIpe’s C++ & Python logs ———
os.environ["GLOG_minloglevel"]    = "3"
os.environ["GLOG_logtostderr"]    = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from absl import logging as _absl_logging
_absl_logging.set_verbosity(_absl_logging.FATAL)
_absl_logging.set_stderrthreshold(_absl_logging.FATAL)

# ——— MediaPipe solutions & styles ———
mp_holistic  = mp.solutions.holistic
mp_hands     = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing   = mp.solutions.drawing_utils
mp_styles    = mp.solutions.drawing_styles

# ——— Instantiate all models ONCE ———
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    refine_face_landmarks=False,  # we’ll run FaceMesh separately
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5
)
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ——— Uniform DrawingSpec for pose (joints+bones) ———
pose_spec = mp_styles.get_default_pose_landmarks_style()

def reproject_landmarks(landmarks, crop_box, frame_shape):
    x1, y1, x2, y2 = crop_box
    W, H = frame_shape[1], frame_shape[0]
    w, h = x2 - x1, y2 - y1
    pts = []
    for lm in landmarks.landmark:
        px = int(lm.x * w) + x1
        py = int(lm.y * h) + y1
        pts.append((px, py))
    return pts

def draw_refined_hands(frame, pose_landmarks):
    h, w = frame.shape[:2]
    wrist_idxs = [15, 16]  # left/right wrist in Holistic
    for wrist_idx in wrist_idxs:
        lm = pose_landmarks.landmark[wrist_idx]
        if lm.visibility < 0.2:
            continue
        cx, cy = int(lm.x * w), int(lm.y * h)
        size = int(0.25 * min(w, h))
        x1, y1 = max(0, cx - size), max(0, cy - size)
        x2, y2 = min(w, cx + size), min(h, cy + size)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)
        if not res.multi_hand_landmarks:
            continue
        pts = reproject_landmarks(res.multi_hand_landmarks[0], (x1, y1, x2, y2), frame.shape)
        for p, q in mp_hands.HAND_CONNECTIONS:
            cv2.line(frame, pts[p], pts[q], (0, 255, 255), 2)
        for px, py in pts:
            cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

def draw_refined_face(frame, face_landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x for lm in face_landmarks.landmark]
    ys = [lm.y for lm in face_landmarks.landmark]
    if not xs or not ys:
        return
    x1, y1 = int(min(xs) * w), int(min(ys) * h)
    x2, y2 = int(max(xs) * w), int(max(ys) * h)
    pad = 20
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        return
    pts = reproject_landmarks(res.multi_face_landmarks[0], (x1, y1, x2, y2), frame.shape)
    for p, q in mp_holistic.FACEMESH_TESSELATION:
        cv2.line(frame, pts[p], pts[q], (0, 255, 0), 1)
    for px, py in pts:
        cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {input_path}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=os.path.basename(input_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hol = holistic.process(rgb)
        annotated = frame.copy()

        # ——— Draw rough Holistic pose with uniform DrawingSpec ———
        # if hol.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         annotated,
        #         hol.pose_landmarks,
        #         mp_holistic.POSE_CONNECTIONS,
        #         landmark_drawing_spec=pose_spec,
        #         connection_drawing_spec=pose_spec,  # uniform, no dict
        #     )

        # ——— Draw rough face mesh tessellation + contours ———
        if hol.face_landmarks:
            tess_style    = mp_styles.get_default_face_mesh_tesselation_style()
            contour_style = mp_styles.get_default_face_mesh_contours_style()
            mp_drawing.draw_landmarks(
                annotated, hol.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                connection_drawing_spec=tess_style
            )
            mp_drawing.draw_landmarks(
                annotated, hol.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                connection_drawing_spec=contour_style
            )

        # ——— Refine hands & face in separate passes ———
        if hol.pose_landmarks:
            draw_refined_hands(annotated, hol.pose_landmarks)
        if hol.face_landmarks:
            draw_refined_face(annotated, hol.face_landmarks)

        out.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

if __name__ == "__main__":
    input_path = "../data/actedgestures/video_01.MP4"
    output_path = "video_01.mp4"
    process_video(input_path, output_path)

    holistic.close()
    hands.close()
    face_mesh.close()