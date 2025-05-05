import cv2
import mediapipe as mp
import numpy as np
import sys

sys.path.append(".")
from enhance.body_description.finger_cls_GNN import est_fingers, FINGER_NAMES
import config.flags as flags

# Thresholds for classification (tunable):
SIDE_X_OFFSET       = 0.15
ABOVE_Y_OFFSET      = 0.2
FAR_ABOVE_Y_OFFSET  = 0.4
DEPTH_FRONT_TH      = -0.05
DEPTH_BACK_TH       = 0.05

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

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


def desc_hand_horizontal(hand_center, center_face, threshold=0.3):
    x, _ = hand_center

    if x < center_face:
        return "left"
    elif x > center_face:
        return "right"
    elif center_face + threshold < x < center_face - threshold:
        return "vertical"
    else:
        return "horizontal position undetermined"


def desc_hand_vertical(hand_center, center_face, threshold=0.3):
    _, y = hand_center
    
    if y < center_face:
        return "above"
    elif y > center_face:
        return "below"
    elif center_face + threshold < y < center_face - threshold:
        return "horizontal"
    else:
        return "vertical position undetermined"


def desc_palm_dir(res, hand_label, threshold=0.3):
    
    hand_landmarks = (
        res.left_hand_landmarks
        if hand_label.lower() == "right"
        else res.right_hand_landmarks
    )
    if not hand_landmarks:
        return None
    
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
        return None
    
    normal /= norm # normalize

    abs_n = np.abs(normal)
    axis = np.argmax(abs_n)
    if abs_n[axis] < threshold:
        return None
    
    # Matrix: axis → [negative direction, positive direction]
    lookup = [
        ["palm facing left",       "palm facing right"],      # X axis
        ["palm facing up",         "palm facing down"],       # Y axis
        ["palm facing the camera", "back facing the camera"], # Z axis
    ]

    sign = int(normal[axis] > 0)
    return lookup[axis][sign]


def desc_face_orientation(face_landmarks, threshold=0.3):
    """
    Estimate which way the face is oriented, using a lookup matrix:
      axis 0 (x): (“turned left”,   “turned right”)
      axis 1 (y): (“tilted up”,     “tilted down”)
      axis 2 (z): (“facing camera”, “turned away”)

    Args:
      face_landmarks: a MediaPipe FaceMesh LandmarkList
      threshold: minimum dominant‐axis magnitude to decide

    Returns:
      One of: "facing camera", "turned away", "tilted up", "tilted down",
              "turned left", "turned right", or None
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
        return None
    
    normal /= norm
    # flip normal to point out of the face
    normal *= -1

    # find the dominant axis and check threshold
    abs_n = np.abs(normal)
    axis = int(np.argmax(abs_n))  # 0=X, 1=Y, 2=Z
    if abs_n[axis] < threshold:
        return None

    # lookup matrix: [axis][sign]
    # sign = 0 if normal[axis] < 0, else 1
    lookup = [
        ("face's turned left", "face's turned right"), # X axis
        ("face's tilted up",   "face's tilted down"),  # Y axis
        ("facing the camera",  "face's turned away"),  # Z axis
    ]
    
    sign = int(normal[axis] > 0)
    return lookup[axis][sign]


def describe_relative_hand(face_center, hand_landmarks):

    if face_center is None or hand_landmarks is None:
        return None

    # xs = [lm.x for lm in hand_landmarks]
    # ys = [lm.y for lm in hand_landmarks]
    # hand_bbox = (min(xs), min(ys), max(xs), max(ys))
    # hx0, hy0, hx1, hy1 = hand_bbox
    # hand_center = ((hx0 + hx1) / 2, (hy0 + hy1) / 2)
    hand_center = (hand_landmarks.x, hand_landmarks.y)
    
    x, y = face_center
    hand_horiz  = desc_hand_horizontal(hand_center, x,)
    hand_vert   = desc_hand_vertical(hand_center, y)
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

    hand_label = hand_label.capitalize()

    if relative_hand_desc is None and palm_desc is None:
        return None

    if relative_hand_desc is None:
        return f"{hand_label} hand's {palm_desc}."

    if palm_desc is None:
        return f"{hand_label} hand is {relative_hand_desc}."

    return f"{hand_label} hand is {relative_hand_desc} with the {palm_desc}."


def desc_hands(res):

    if not res.pose_landmarks:
        return None
    face_pose = res.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
    if not face_pose:
        return None
    
    face_center = (face_pose.x, face_pose.y)
    
    if not res:
        return None
    
    hands_list = ["Left", "Right"]

    hands_desc = []
    for hand_label in hands_list:
        
        wrist = res.pose_landmarks.landmark[(
            mp_pose.PoseLandmark.LEFT_WRIST
            if hand_label.lower() == "right"
            else mp_pose.PoseLandmark.RIGHT_WRIST
        )]
        if wrist is None:
            hands_desc.append(f"{hand_label.capitalize()} hand isn't visible.")
            continue
        relative_hand_desc = describe_relative_hand(face_center, wrist)
        
        palm_desc = desc_palm_dir(res, hand_label.lower()) if flags.describe_hands else None
        # skip if no palm direction and no relative hand description
        if relative_hand_desc is None and palm_desc is None:
            continue

        hand_desc = formulate_desc(relative_hand_desc, palm_desc, hand_label)
        
        # fingers_desc = desc_fingers(hand_lms.landmark)
        # full_hand_desc = f"{hand_desc} {fingers_desc}" if fingers_desc else hand_desc
        # hands_desc.append(full_hand_desc)

        # Separate hand descriptions
        hands_desc.append(hand_desc)
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


def desc_person(res) -> tuple:
    """Process a frame and describe the face and hands.

    Args:
        res: The MediaPipe Holistic object containing pose and hand landmarks.

    Returns:
        A list of descriptions for the face and hands.
    """
    descriptions = []
    
    if not res:
        return ["No person detected."]

    # face_bbox = get_face_bbox(face_pose)

    # Describe face
    face_dir = desc_face_orientation(res.face_landmarks)
    descriptions.append(f"They're {face_dir}.") if face_dir else None

    # Describe hands
    hand_desc = desc_hands(res)
    descriptions += hand_desc if hand_desc else ["No hands detected."]

    return descriptions