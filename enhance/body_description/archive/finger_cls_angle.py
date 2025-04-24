import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Joint indices
# Thumb: CMC(1), MCP(2), IP(3), TIP(4)
# Non-thumb fingers: MCP, PIP, DIP, TIP
FINGER_JOINTS = {
    "Thumb":  [1, 2, 3, 4],
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10,11,12],
    "Ring":   [13,14,15,16],
    "Pinky":  [17,18,19,20]
}

def angle_3d(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

# Threshold for straight (raised) fingers
ANGLE_THRESH = 160.0
THUMB_ANGLE_THRESH = 130.0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res     = hands.process(img_rgb)

    if res.multi_hand_landmarks:
        hand_lms = res.multi_hand_landmarks[0]
        # normalized landmarks
        lms = [(lm.x, lm.y, lm.z) for lm in hand_lms.landmark]
        wrist = lms[0]

        statuses = {}
        for name, idxs in FINGER_JOINTS.items():
            coords = [lms[i] for i in idxs]
            if name == "Thumb":
                # Thumb: angle at MCP (CMC->MCP->IP), angle at IP (MCP->IP->TIP)
                ang1 = angle_3d(coords[0], coords[1], coords[2])
                ang2 = angle_3d(coords[1], coords[2], coords[3])
                up = (ang1 > ANGLE_THRESH and ang2 > THUMB_ANGLE_THRESH)
            else:
                # Non-thumb: angle at MCP (wrist->MCP->PIP), PIP, DIP
                ang0 = angle_3d(wrist, coords[0], coords[1])
                ang1 = angle_3d(coords[0], coords[1], coords[2])
                ang2 = angle_3d(coords[1], coords[2], coords[3])
                up = (ang0 > ANGLE_THRESH and ang1 > ANGLE_THRESH and ang2 > ANGLE_THRESH)
            statuses[name] = up

        # Draw skeleton
        mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

        # Annotate all four joints
        h, w, _ = frame.shape
        for name, idxs in FINGER_JOINTS.items():
            for i, idx in enumerate(idxs):
                x, y, z = lms[idx]
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
                label = ["A", "B", "C", "TIP"][i]
                cv2.putText(frame, f"{name}_{label}", (cx + 2, cy - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Overlay raised fingers
        raised = [n for n, up in statuses.items() if up]
        text = "Raised: " + (", ".join(raised) if raised else "None")
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No hand detected", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("4-Joint Finger Detection with Wrist", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
