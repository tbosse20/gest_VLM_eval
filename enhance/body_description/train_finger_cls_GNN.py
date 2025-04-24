import os
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GCNConv
from torch_geometric.nn import global_mean_pool
import numpy as np

# ─── Skeleton edges for MediaPipe 21‐point hand ───
SKELETON_EDGES = [
    (0,  1), ( 1,  2), ( 2,  3), ( 3,  4),
    (0,  5), ( 5,  6), ( 6,  7), ( 7,  8),
    (0,  9), ( 9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
edge_index = torch.tensor(SKELETON_EDGES, dtype=torch.long).t().contiguous()

# ─── Map gesture names to 5‐dim finger vectors ───
gesture_to_fingers = {
    "call":             [1,0,0,0,1],
    "dislike":          [1,0,0,0,0],
    "fist":             [0,0,0,0,0],
    "four":             [0,1,1,1,1],
    "like":             [1,0,0,0,0],
    "mute":             [0,1,0,0,0],
    "ok":               [1,1,0,0,0],
    "one":              [0,1,0,0,0],
    "palm":             [1,1,1,1,1],
    "peace":            [0,1,1,0,0],
    "peace_inverted":   [0,1,1,0,0],
    "rock":             [0,1,0,0,1],
    "stop":             [1,1,1,1,1],
    "stop_inverted":    [1,1,1,1,1],
    "three":            [0,1,1,1,0],
    "three2":           [1,1,1,0,0],
    "two_up":           [0,1,1,0,0],
    "two_up_inverted":  [0,1,1,0,0]
}


# ─── GNN model ───
class HandGNN(torch.nn.Module):
    def __init__(self, in_channels=4, hidden=64, num_labels=5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.bn1   = BatchNorm(hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.bn2   = BatchNorm(hidden)
        self.lin   = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden//2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden//2, num_labels)
        )
    def forward(self, data):
        x, edge_idx, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.bn1(self.conv1(x, edge_idx)))
        x = F.relu(self.bn2(self.conv2(x, edge_idx)))
        x = global_mean_pool(x, batch)
        return self.lin(x)


def normalize_landmarks(lm2d):
    """Center on wrist and scale by distance wrist->middle_mcp."""
    
    pts = np.array(lm2d, dtype=np.float32)
    wrist = pts[0]
    pts -= wrist
    
    # middle_mcp is index 9
    scale = np.linalg.norm(pts[9]) + 1e-6
    pts /= scale
    return pts

def augment_landmarks(pts):
    """Random rotation, scaling and jitter."""
    # scale
    scale = np.random.uniform(0.8, 1.2)  # (80% to 120%)
    pts *= scale
    
    # rotation
    theta = np.random.uniform(-np.pi, np.pi)  # (–180° to +180°)
    c, s  = np.cos(theta), np.sin(theta)
    R     = np.array([[c, -s],[s, c]])
    xy    = pts[:, :2].dot(R.T)
    
    # jitter
    jitter = np.random.normal(scale=0.01, size=xy.shape)
    xy += jitter
    pts[:, :2] = xy
    
    return pts

def load_and_process_landmarks(entry, augment=False):
    
    # landmarks is a list of lists; take the first sub‐list
    lm = entry["landmarks"][0]
    
    # skip empty or malformed entries
    if not lm or len(lm) != 21 or any(len(p) != 2 for p in lm):
        # print(f"Invalid landmarks in entry: {lm}")
        return None
    
    # normalize and optionally augment
    pts = normalize_landmarks(lm)
    if augment:
        pts = augment_landmarks(pts)
        
    return pts

def process_sample(entry, augment=False):
            
    # labels is a list of strings; we take the first
    gesture = entry["labels"][0]
    if gesture not in gesture_to_fingers:
        # print(f"Gesture '{gesture}' not in gesture_to_fingers.")
        return None
    
    pts = load_and_process_landmarks(entry, augment=augment)
    if pts is None:
        return None
    x = torch.tensor(pts, dtype=torch.float)   # shape [21,2]
    
    # add bone vector to parent as extra features
    bone_feats = []
    parent = {**{1:0}, **{i:i-1 for i in range(2,21)}}
    for i in range(21):
        px, py, = pts[parent.get(i,0),0], pts[parent.get(i,0),1]
        bone_feats.append([pts[i,0]-px, pts[i,1]-py])
    bone_feats = torch.tensor(bone_feats, dtype=torch.float)
    x = torch.cat([x, bone_feats], dim=1)  # [21,4]
    y = torch.tensor([gesture_to_fingers[gesture]], dtype=torch.float)  # [1,5]
    
    return Data(x=x, edge_index=edge_index, y=y)
    
def load_data_from_dir(directory: str, augment:bool = False, cutoff: int = 0):
    """ Load data from a directory of JSON files.
    Each file is a dictionary of entries, where each entry is a list of landmarks.
    
    Args:
        directory (str): Path to the directory containing JSON files.
        augment (bool): Whether to apply data augmentation.
        cutoff (int): Maximum number of samples to load from each file.
        
    Returns:
        list: List of Data objects.
    """
    
    if not os.path.isdir(directory):
        raise ValueError(f"Directory '{directory}' does not exist.")
    
    data_list = []
    base_name = os.path.basename(directory)
    for fname in tqdm(os.listdir(directory), desc=f"Loading '{base_name}'"):
        if not fname.lower().endswith(".json"):
            continue
        path = os.path.join(directory, fname)
        with open(path, "r") as f:
            js = json.load(f)
        # js is a dict: each value is one annotation entry
        
        data_sub_list = []
        for entry in js.values():
            
            # process_sample returns None if the entry is invalid
            data_sample = process_sample(entry, augment=augment)
            if data_sample is None:
                continue
            data_sub_list.append(data_sample)
            
            if cutoff > 0 and len(data_sub_list) >= cutoff:
                break
        data_list += data_sub_list
    
    return data_list
    
if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="Train a GNN model for hand gesture classification.")
    parser.add_argument("--sanity", action="store_true", help="Run sanity check on the model.")
    args = parser.parse_args()

    # Shuffle and split train/val
    from sklearn.model_selection import train_test_split

    # ─── Paths ───
    BASE_DIR      = "../data/HaGRID"
    TRAIN_VAL_DIR = os.path.join(BASE_DIR, "ann_train_val")
    TEST_DIR      = os.path.join(BASE_DIR, "ann_test")

    # ─── Hyperparameters ───
    BATCH_SIZE  = 32    if not args.sanity else 4
    EPOCHS      = 4     if not args.sanity else 1
    LR          = 1e-3
    TRAIN_SPLIT = 0.8
    
    # ─── Load datasets ───
    cut_off = 0 if not args.sanity else 1 # Sanity check: load only 4 samples of each gesture
    train_val = load_data_from_dir(TRAIN_VAL_DIR, augment=True, cutoff=cut_off)
    test_data = load_data_from_dir(TEST_DIR,      augment=True, cutoff=cut_off)

    # After loading all_train_val:
    train_data, val_data = train_test_split(
        train_val,
        train_size=TRAIN_SPLIT,
        random_state=42,
        shuffle=True
    )
    print(f"{len(train_data):,} train samples")
    print(f"{len(val_data):,}   val samples")
    print(f"{len(test_data):,}  test samples")
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE)

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = HandGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.BCEWithLogitsLoss()

    # ─── Training & evaluation ───
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Train Epoch {epoch:02d}"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
        scheduler.step()

        # eval
        model.eval()
        correct = total = 0
        for loader, name in [(train_loader,"Train"), (val_loader,"Val")]:
            c = t = 0
            for batch in loader:
                batch = batch.to(device)
                pred = (torch.sigmoid(model(batch))>0.5).float()
                c += (pred==batch.y).sum().item()
                t += batch.y.numel()
            acc = c/t*100
            print(f"Epoch {epoch:02d} | {name} Acc: {acc:5.1f}% |", end=" ")
        print()
        
        # ─── Save model weights ───
        torch.save(model.state_dict(), "enhance/body_description/hand_gnn.pth")
        print("Model saved to hand_gnn.pth")

    # ─── Test ───
    model.eval()
    c = t = 0
    for batch in test_loader:
        batch = batch.to(device)
        pred = (torch.sigmoid(model(batch))>0.5).float()
        c += (pred==batch.y).sum().item()
        t += batch.y.numel()
    print(f"Test Acc: {c/t*100:.1f}%")