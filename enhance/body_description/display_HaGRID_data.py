import cv2
import torch
import matplotlib.pyplot as plt
import sys, os
import json

sys.path.append(".")
from enhance.body_description.train_finger_cls_GNN import process_sample

BASE_DIR      = "./enhance/HaGRID"  # adjust as needed
TRAIN_VAL_DIR = os.path.join(BASE_DIR, "ann_train_val")

def load_single_sample(directory: str, gesture: int = 0, idx: int = 0):
    
    directories = os.listdir(directory)
    if len(directories) == 0:
        raise ValueError(f"No directories found in {directory}.")
    
    fname = directories[gesture]
    if not fname.lower().endswith(".json"):
        raise ValueError(f"File {fname} is not a JSON file.")
        
    path = os.path.join(directory, fname)
    with open(path, "r") as f:
        js = json.load(f)
    if len(js) == 0:
        raise ValueError(f"No entries found in {fname}.")
    
    entry = list(js.values())[idx]  # Get the first entry
    data_sample = process_sample(entry, augment=True)
    if data_sample is None:
        raise ValueError(f"Invalid data sample at index '{idx}' in '{fname}'.")

    return data_sample  # Return the first valid sample

# Load raw sample from json file
data_sample = load_single_sample(TRAIN_VAL_DIR, 2, 3)

# Plot the first sample
def plot_hand(data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Convert to numpy for plotting
    x = data.x.numpy()
    edge_index = data.edge_index.numpy()
    gesture = data.y.numpy()

    # Plot nodes
    ax.scatter(x[:, 0], x[:, 1], c='blue', s=100)

    # Plot edges
    for i in range(edge_index.shape[1]):
        start = x[edge_index[0, i]]
        end = x[edge_index[1, i]]
        ax.plot([start[0], end[0]], [start[1], end[1]], c='red')

    # Set limits and labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Hand Skeleton {gesture}')

    plt.show()
plot_hand(data_sample)