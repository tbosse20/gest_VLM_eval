import cv2
import torch
import mediapipe as mp
import numpy as np
from torch_geometric.data import Data
import sys
sys.path.append(".")
from enhance.body_description.train_finger_cls_GNN import HandGNN, SKELETON_EDGES

# Build edge_index tensor
edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()

# Finger names
FINGER_NAMES = ["Thumb","Index","Middle","Ring","Pinky"]

# Device and model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandGNN(in_channels=4).to(device)
model.load_state_dict(torch.load("hand_gnn.pth", map_location=device))
model.eval()

# MediaPipe Hands initialization
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Parent index mapping for bone vectors
parent = {1:0}
parent.update({i: i-1 for i in range(2,21)})

def est_fingers(lm):
    """Estimate finger positions based on 2D landmarks."""
    
    if len(lm) != 21:
        raise ValueError("Expected 21 landmarks, got {}".format(len(lm)))

    # Extract 2D landmarks and normalize
    lm2d = np.array([(l.x, l.y) for l in lm], dtype=np.float32)  # (21,2)
    # Center on wrist and scale by distance to middle_mcp (index 9)
    wrist = lm2d[0]
    lm2d_centered = lm2d - wrist
    scale = np.linalg.norm(lm2d_centered[9]) + 1e-6
    lm_norm = lm2d_centered / scale

    # Compute bone vectors as additional features
    bone_feats = []
    for i in range(21):
        px, py = lm_norm[parent.get(i,0)]
        bone_feats.append([lm_norm[i,0] - px, lm_norm[i,1] - py])
    bone_feats = np.array(bone_feats, dtype=np.float32)

    # Combine into node feature tensor of shape (21,4)
    x_input = np.hstack([lm_norm, bone_feats])
    x_tensor = torch.tensor(x_input, dtype=torch.float).to(device)

    # Prepare PyG Data object
    batch = torch.zeros(x_tensor.size(0), dtype=torch.long).to(device)
    data = Data(x=x_tensor, edge_index=edge_index.to(device), batch=batch)

    # Inference
    with torch.no_grad():
        out = model(data)  # (1,5)
        pred = (torch.sigmoid(out) > 0.5).cpu().numpy()[0]
        
    return pred

if __name__ == "__main__":
    # Webcam loop
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_res = hands.process(img_rgb)

        if hands_res.multi_hand_landmarks:
            h, w, _ = frame.shape
            
            # Estimate raised fingers
            lm = hands_res.multi_hand_landmarks[0].landmark
            pred = est_fingers(lm)

            # Draw skeleton
            mp_draw.draw_landmarks(frame, hands_res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            # Overlay predictions
            raised = [name for i, name in enumerate(FINGER_NAMES) if pred[i]]
            text = "Raised: " + (", ".join(raised) if raised else "None")
            cv2.rectangle(frame, (0,0), (w,30), (0,0,0), -1)
            cv2.putText(frame, text, (10,20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        else:
            cv2.putText(frame, "No hand detected", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow("Live Finger Inference (Normalized + Bone Feats)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
