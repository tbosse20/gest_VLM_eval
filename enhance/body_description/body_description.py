import cv2
import mediapipe as mp
import numpy as np
import sys

sys.path.append(".")
from enhance.body_description.finger_cls_GNN import est_fingers, FINGER_NAMES

# Thresholds for classification (tunable):
SIDE_X_OFFSET       = 0.15
ABOVE_Y_OFFSET      = 0.2
FAR_ABOVE_Y_OFFSET  = 0.4
DEPTH_FRONT_TH      = -0.05
DEPTH_BACK_TH       = 0.05

mp_hands = mp.solutions.hands

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
    print(ratio)
    
    if ratio > 0.5 + rel_threshold:
        return "in front of"
    elif ratio < 0.5 - rel_threshold:
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
    wrist    = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    idx_mcp  = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    pinky_mcp= hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

    v1 = np.array([idx_mcp.x - wrist.x,
                   idx_mcp.y - wrist.y,
                   idx_mcp.z - wrist.z])
    v2 = np.array([pinky_mcp.x - wrist.x,
                   pinky_mcp.y - wrist.y,
                   pinky_mcp.z - wrist.z])
    
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
        ["palm facing left",       "palm facing right"],      # X axis
        ["palm facing up",         "palm facing down"],       # Y axis
        ["palm facing the camera", "back facing the camera"], # Z axis
    ]

    sign = int(normal[axis] <= 0)
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
        ("face's turned left", "face's turned right"), # X axis
        ("face's tilted up",   "face's tilted down"),  # Y axis
        ("facing the camera",  "face's turned away"),  # Z axis
    ]
    sign = int(normal[axis] > 0)
    return lookup[axis][sign]


def describe_relative_hand(face_bbox, hand_landmarks):

    if face_bbox is None or hand_landmarks is None:
        return None

    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]
    hand_bbox = (min(xs), min(ys), max(xs), max(ys))
    hx0, hy0, hx1, hy1 = hand_bbox
    hand_center = ((hx0 + hx1) / 2, (hy0 + hy1) / 2)
    
    xmin, ymin, xmax, ymax = face_bbox
    hand_horiz  = desc_hand_horizontal(hand_center, xmin, xmax)
    hand_vert   = desc_hand_vertical(hand_center, ymin, ymax)
    # hand_depth  = desc_hand_depth(hand_bbox, face_bbox)
    # hand_position_desc = f"{hand_horiz}, {hand_vert}, and {hand_depth} their face"
    hand_position_desc = f"{hand_horiz} and {hand_vert} their face"

    return hand_position_desc


def get_face_bbox(face_lms):

    if not face_lms:
        return None

    # Get the bounding box of the face landmarks x, y coordinates
    xs = [lm.x for lm in face_lms.landmark]
    ys = [lm.y for lm in face_lms.landmark]
    face_bbox = (min(xs), min(ys), max(xs), max(ys))

    return face_bbox


def formulate_desc(relative_hand_desc, palm_desc, hand_label):

    Hand_label = hand_label.capitalize()

    if relative_hand_desc is None and palm_desc is None:
        return None

    if relative_hand_desc is None:
        return f"{Hand_label} hand's {palm_desc}."

    if palm_desc is None:
        return f"{Hand_label} hand {relative_hand_desc}."

    return f"{Hand_label} hand " + f"is {relative_hand_desc} with the {palm_desc}."


def desc_hands(hands_list, face_bbox):

    if not face_bbox or not hands_list:
        return None

    hands_desc = []
    for hand_lms, hand_label in hands_list:
        
        if hand_lms is None:
            hands_desc.append(f"{hand_label.capitalize()} hand isn't visible.")
            continue

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
        return "Making a fist."

    desc = "Raised fingers are: " + ", ".join(raised) + "."

    return desc


def write_desc(frame, descriptions, pos=(10, 30), font_scale=0.5, color=(255, 255, 255)):
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


def desc_person(face_res, hands_list) -> tuple:
    """Process a frame and describe the face and hands.

    Args:
        face_res: The MediaPipe FaceMesh object.
        hands_list: The MediaPipe Hands object.

    Returns:
        A list of descriptions for the face and hands.
    """
    descriptions = []

    face_bbox = get_face_bbox(face_res)

    # Describe face
    face_dir = desc_face_orientation(face_res)
    descriptions.append(f"They're {face_dir}.")

    # Describe hands
    hand_desc = desc_hands(hands_list, face_bbox)
    descriptions += hand_desc if hand_desc else ["No hands detected."]

    return descriptions