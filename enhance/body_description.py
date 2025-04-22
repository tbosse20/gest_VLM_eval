import cv2
import mediapipe as mp
import os
import numpy as np

# Initialize MediaPipe Face Mesh and Hands.
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Thresholds for classification (tunable):
SIDE_X_OFFSET = 0.15
ABOVE_Y_OFFSET = 0.2
FAR_ABOVE_Y_OFFSET = 0.4
DEPTH_FRONT_TH = -0.05
DEPTH_BACK_TH = 0.05


def depth_desc(hand_bbox, face_bbox, rel_threshold=0.2):
    """
    Estimate depth by comparing hand vs. face bbox area.

    Args:
      hand_bbox: (xmin, ymin, xmax, ymax) for the hand
      face_bbox: (xmin, ymin, xmax, ymax) for the face
      rel_threshold: fraction above/below 1.0 to call front/behind

    Returns:
      "in front of the face" if hand appears rel_threshold×larger than face,
      "behind the face" if hand appears rel_threshold×smaller than face,
      otherwise "at same depth as the face".
    """
    # compute areas
    hx0, hy0, hx1, hy1 = hand_bbox
    fx0, fy0, fx1, fy1 = face_bbox

    hand_area = max(0, (hx1 - hx0)) * max(0, (hy1 - hy0))
    face_area = max(0, (fx1 - fx0)) * max(0, (fy1 - fy0))
    if face_area == 0:
        return "depth undetermined"

    ratio = hand_area / face_area

    if ratio > 1 + rel_threshold:
        return "in front of"
    elif ratio < 1 - rel_threshold:
        return "behind"
    else:
        return "beside"


def horizontal_desc(hand_center, xmin, xmax):
    x, _ = hand_center
    left = xmin
    right = xmax

    if x < left:
        return "left"
    elif x > right:
        return "right"
    elif xmin < x < xmax:
        return "centered"
    else:
        return "horizontal position undetermined"


def vertical_desc(hand_center, ymin, ymax):
    _, y = hand_center
    top = ymax - ABOVE_Y_OFFSET
    bottom = ymin

    if y < top:
        return "above"
    elif y > bottom:
        return "below"
    elif ymin < y < ymax:
        return "centered"
    else:
        return "vertical position undetermined"


def get_palm_direction(hand_landmarks, hand_label, threshold=0.3):
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    idx_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    v1 = np.array([idx_mcp.x - wrist.x, idx_mcp.y - wrist.y, idx_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y, pinky_mcp.z - wrist.z])

    normal = np.cross(v2, v1)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return "orientation undetermined"
    normal /= norm
    # flip normal to point out of the hand
    normal *= -1 if hand_label == "left" else 1

    abs_n = np.abs(normal)
    axis = np.argmax(abs_n)
    if abs_n[axis] < threshold:
        return "orientation unclear"

    if axis == 2:  # Z axis dominates
        return "palm facing the camera" if normal[2] < 0 else "back facing the camera"
    if axis == 1:  # Y axis
        return "palm facing down" if normal[1] > 0 else "palm facing up"
    # X axis
    return "palm facing right" if normal[0] > 0 else "palm facing left"


def get_face_direction(face_landmarks, threshold=0.3):
    """
    Estimate which way the face is oriented, using a lookup matrix:
      axis 0 (x): (“turned left”,     “turned right”)
      axis 1 (y): (“tilted up”,       “tilted down”)
      axis 2 (z): (“facing camera”,   “turned away”)

    Args:
      face_landmarks: a MediaPipe FaceMesh LandmarkList
      threshold: minimum dominant‐axis magnitude to decide

    Returns:
      One of: "facing camera", "turned away", "tilted up", "tilted down",
              "turned left", "turned right", or "orientation undetermined"
    """
    # pick four facial landmarks
    LE, RE, NT, CH = 33, 263, 1, 152
    le = face_landmarks.landmark[LE]
    re = face_landmarks.landmark[RE]
    nt = face_landmarks.landmark[NT]
    ch = face_landmarks.landmark[CH]

    # two vectors in the face plane
    v1 = np.array([re.x - le.x, re.y - le.y, re.z - le.z])
    v2 = np.array([ch.x - nt.x, ch.y - nt.y, ch.z - nt.z])

    # normal vector
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return "orientation undetermined"
    normal /= norm
    # flip normal to point out of the face
    normal *= -1

    # find the dominant axis and check threshold
    abs_n = np.abs(normal)
    axis = int(np.argmax(abs_n))  # 0=X, 1=Y, 2=Z
    if abs_n[axis] < threshold:
        return "orientation unclear"

    # lookup matrix: [axis][sign]
    # sign = 0 if normal[axis] < 0, else 1
    lookup = [
        ("turned left", "turned right"),  # X axis
        ("tilted up", "tilted down"),  # Y axis
        ("facing camera", "turned away"),  # Z axis
    ]
    sign = int(normal[axis] > 0)
    return lookup[axis][sign]


def describe_hand(face_bbox, face_center_z, hand_landmarks, hand_label):
    xmin, ymin, xmax, ymax = face_bbox
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    xmin, ymin, xmax, ymax = face_bbox

    # 1) compute hand bbox from all landmarks
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    hand_bbox = (min(xs), min(ys), max(xs), max(ys))

    # Get center of hand x and y coordinates
    hx0, hy0, hx1, hy1 = hand_bbox
    hand_center = ((hx0 + hx1) / 2, (hy0 + hy1) / 2)

    # 2) depth via bbox‐area comparison
    hand_horiz = horizontal_desc(hand_center, xmin, xmax)
    hand_vert = vertical_desc(hand_center, ymin, ymax)
    hand_depth = depth_desc(hand_bbox, face_bbox)
    hand_position_desc = (
        f"{hand_label.capitalize()} hand is {hand_horiz}, {hand_vert}, and {hand_depth} "
        f"of their face."
    )

    palm_dir = get_palm_direction(hand_landmarks, hand_label.lower())
    # assemble into one natural‐language sentence:
    return f"{hand_position_desc} with the {palm_dir}."


def main(video_path):
    if video_path:
        if not video_path.endswith((".mp4", ".avi", ".mov", ".MP4")):
            raise ValueError("Video path must end with .mp4")
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file {video_path} not found")
    else:
        video_path = 0
    cap = cv2.VideoCapture(video_path)

    if video_path != 0:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_name = os.path.basename(video_path)
        out = cv2.VideoWriter(video_name, fourcc, fps, (W, H))

    with mp_face.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh, mp_hands.Hands(
        min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2
    ) as hands:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_res = face_mesh.process(rgb)
            hands_res = hands.process(rgb)

            descriptions = []

            # If face detected
            if face_res.multi_face_landmarks:
                face_lms = face_res.multi_face_landmarks[0]
                xs = [lm.x for lm in face_lms.landmark]
                ys = [lm.y for lm in face_lms.landmark]
                zs = [lm.z for lm in face_lms.landmark if abs(lm.x - 0.5) < 0.1]
                face_bbox = (min(xs), min(ys), max(xs), max(ys))
                face_center_z = sum(zs) / len(zs) if zs else 0.0

                mp_drawing.draw_landmarks(
                    frame,
                    face_lms,
                    mp_face.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1
                    ),
                )

                face_dir = get_face_direction(face_lms)
                descriptions.append(f"Face is {face_dir}.")

                # Describe each hand relative to that face
                if hands_res.multi_hand_landmarks:
                    for hand_lms, hand_h in zip(
                        hands_res.multi_hand_landmarks, hands_res.multi_handedness
                    ):
                        hand_label = hand_h.classification[0].label
                        desc = describe_hand(
                            face_bbox, face_center_z, hand_lms, hand_label
                        )
                        descriptions.append(desc)

                        mp_drawing.draw_landmarks(
                            frame, hand_lms, mp_hands.HAND_CONNECTIONS
                        )

            # Overlay
            for i, d in enumerate(descriptions):
                cv2.putText(
                    frame,
                    d,
                    (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

            out.write(frame) if video_path != 0 else None
            # cv2.imshow("Pose Description", frame)
            # if cv2.waitKey(5) & 0xFF == 27:
            #     break

    cap.release()
    out.release() if video_path != 0 else None
    cv2.destroyAllWindows()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="MediaPipe Pose Description")
    parser.add_argument(
        "video_path",
        type=str,
        default=None,
        help="Path to video file (default: webcam)",
    )
    args = parser.parse_args()

    main(args.video_path)
