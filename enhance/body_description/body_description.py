import cv2
import mediapipe as mp
import numpy as np
import sys

sys.path.append(".")
from enhance.body_description.finger_cls_GNN import est_fingers, FINGER_NAMES
import enhance.util as util

# Initialize MediaPipe Face Mesh and Hands.
mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Thresholds for classification (tunable):
SIDE_X_OFFSET = 0.15
ABOVE_Y_OFFSET = 0.2
FAR_ABOVE_Y_OFFSET = 0.4
DEPTH_FRONT_TH = -0.05
DEPTH_BACK_TH = 0.05


def desc_hand_depth(hand_bbox, face_bbox, rel_threshold=0.2):
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


def desc_hand_horizontal(hand_center, left_face, right_face):
    x, _ = hand_center
    left = left_face
    right = right_face

    if x < left:
        return "left"
    elif x > right:
        return "right"
    elif left_face < x < right_face:
        return "vertical"
    else:
        return "horizontal position undetermined"


def desc_hand_vertical(hand_center, upper_face, lower_face):
    _, y = hand_center

    if y < upper_face:
        return "above"
    elif y > lower_face:
        return "below"
    elif upper_face < y < lower_face:
        return "horizontal"
    else:
        return "vertical position undetermined"


def desc_palm_dir(hand_landmarks, hand_label, threshold=0.3):
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

    # Matrix: axis → [negative direction, positive direction]
    lookup = [
        ["palm is facing left", "palm is facing right"],  # X axis
        ["palm is facing up", "palm is facing down"],  # Y axis
        ["palm is facing the camera", "back is facing the camera"],  # Z axis
    ]

    sign = int(normal[axis] > 0)
    return lookup[axis][sign]


def desc_face_orientation(face_landmarks, threshold=0.3):
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

    if not face_landmarks:
        return None

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


def describe_relative_hand(face_bbox, hand_landmarks):

    if face_bbox is None:
        return None

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
    hand_horiz = desc_hand_horizontal(hand_center, xmin, xmax)
    hand_vert = desc_hand_vertical(hand_center, ymin, ymax)
    hand_depth = desc_hand_depth(hand_bbox, face_bbox)
    hand_position_desc = f"{hand_horiz}, {hand_vert}, and {hand_depth} their face"

    return hand_position_desc


def process_face(face_res):

    # If face detected
    if not face_res.multi_face_landmarks:
        return None, None

    face_lms = face_res.multi_face_landmarks[0]
    xs = [lm.x for lm in face_lms.landmark]
    ys = [lm.y for lm in face_lms.landmark]
    zs = [lm.z for lm in face_lms.landmark if abs(lm.x - 0.5) < 0.1]
    face_bbox = (min(xs), min(ys), max(xs), max(ys))

    return face_lms, face_bbox


def formulate_desc(relative_hand_desc, palm_desc, hand_label):

    Hand_label = hand_label.capitalize()

    if relative_hand_desc is None and palm_desc is None:
        return None

    if relative_hand_desc is None:
        return f"{Hand_label} hand's {palm_desc}."

    if palm_desc is None:
        return f"{Hand_label} hand {relative_hand_desc}."

    return f"{Hand_label} hand " + f"is {relative_hand_desc}. " + f"The {palm_desc}."


def desc_hands(hands_res, face_bbox):

    # Describe each hand relative to that face
    if not hands_res.multi_hand_landmarks:
        return None

    hands_desc = []

    for hand_lms, hand_h in zip(
        hands_res.multi_hand_landmarks, hands_res.multi_handedness
    ):
        hand_label = hand_h.classification[0].label

        relative_hand_desc = describe_relative_hand(face_bbox, hand_lms)
        palm_desc = desc_palm_dir(hand_lms, hand_label.lower())

        # skip if no palm direction and no relative hand description
        if relative_hand_desc is None and palm_desc is None:
            continue

        hand_desc = formulate_desc(relative_hand_desc, palm_desc, hand_label)
        fingers_desc = desc_fingers(hand_lms.landmark)
        full_hand_desc = f"{hand_desc} {fingers_desc}" if fingers_desc else hand_desc
        hands_desc.append(full_hand_desc)

        # Separate hand descriptions
        # hands_desc.append(hand_desc)
        # hands_desc.append(fingers_desc)

    return hands_desc


def desc_fingers(hand_lms):
    pred_fingers = est_fingers(hand_lms)
    if pred_fingers is None:
        return None

    # Describe which fingers are raised
    raised = [name for i, name in enumerate(FINGER_NAMES) if pred_fingers[i]]
    if len(raised) == 0:
        return "Fist"

    desc = "Raised fingers: " + ", ".join(raised) + ". "

    return desc


def draw_face(frame, face_lms):

    if not face_lms:
        return

    mp_drawing.draw_landmarks(
        frame,
        face_lms,
        mp_face.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
    )


def draw_hands(frame, hands_res) -> None:
    if not hands_res.multi_hand_landmarks:
        return

    for hand_lms in hands_res.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            hand_lms,
            mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0), thickness=1
            ),
        )


def draw_landmarks(frame, face_lms, hands_res) -> None:
    draw_face(frame, face_lms)
    draw_hands(frame, hands_res)


def draw_desc(frame, descriptions, pos=(10, 30), font_scale=0.5, color=(255, 255, 255)):
    """Draw text on the frame."""
    if not descriptions:
        return frame

    # Overlay text
    for i, d in enumerate(descriptions):
        cv2.putText(
            frame,
            d,
            (pos[0], pos[1] + i * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            1,
            cv2.LINE_AA,
        )
    return frame


def desc_person(frame, draw: int = 0) -> tuple:
    """Process a frame and describe the face and hands.

    Args:
        frame: The input video frame.
        face_mesh: The MediaPipe FaceMesh object.
        hands: The MediaPipe Hands object.
        draw: The drawing level (0: no drawing, 1: draw landmarks, 2: draw text).

    Returns:
        A tuple containing the descriptions and the processed frame.
    """

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process face and hands
    face_res = face_mesh.process(rgb)
    hands_res = hands.process(rgb)

    face_lms, face_bbox = process_face(face_res)
    draw_landmarks(frame, face_lms, hands_res) if draw > 0 else None

    descriptions = []
    # Describe face
    face_dir = desc_face_orientation(face_lms)
    descriptions.append(f"Face is {face_dir}.")

    # Describe hands
    hand_desc = desc_hands(hands_res, face_bbox)
    descriptions += hand_desc if hand_desc else ["No hands detected."]
    draw_desc(frame, descriptions) if draw > 1 else None

    return descriptions, frame


def main(frame, draw: bool = False) -> None:
    """Main function to process video and describe body parts."""

    _, frame = desc_person(frame, draw=draw)

    return frame


if __name__ == "__main__":
    util.main(main)