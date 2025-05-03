# %%
import sys
sys.path.append(".")
import analysis.plot_matrix as plot_matrix
import config.directories as directories

metrics_folder=directories.OUTPUT_FOLDER_PATH
data = plot_matrix.post_process_csv_folder(metrics_folder)

# %%
def sample_correctness(data, correct):
    for df in data:
        
        model_name = df["model_name"].unique()[0]
        
        df = df[df["correct"] == correct]
        df = df[["pred_id", "gt_id", "video_name"]].copy()
        
        # Filter
        df = df[~df["gt_id"].isin([10, 0])]

        # Sort by video name and frame index
        df = df.sort_values(by=["video_name"])
        
        print(f"{'#'*10} {model_name.upper()} {'#'*10}")
        print(df)
        print()
        
sample_correctness(data, correct=True)

# %%
from collections import defaultdict
import pandas as pd

def count_correctness_videos(data):
    """Count total correct predictions per video across all methods, including gt_id."""

    # Initialize a dictionary to count correct predictions and store gt_id
    video_counts = defaultdict(lambda: {"correct": 0, "gt_id": None})

    for df in data:
        for _, row in df.iterrows():
            video = row["video_name"]
            if not row["correct"]:
                continue

            video_counts[video]["correct"] += 1
            
            # Store gt_id if not already set
            if video_counts[video]["gt_id"] is None:
                video_counts[video]["gt_id"] = row["gt_id"]
            elif video_counts[video]["gt_id"] != row["gt_id"]:
                # Optional: warn if conflicting gt_ids for the same video
                print(f"Warning: Conflicting gt_ids for video {video}")

    # Convert to DataFrame
    result_df = pd.DataFrame([
        {"video_name": video, "correct": counts["correct"], "gt_id": counts["gt_id"]}
        for video, counts in video_counts.items()
    ])

    # Filter videos with fewer than 2 correct predictions
    result_df = result_df[result_df["correct"] >= 2]

    # Sort by video name
    result_df = result_df.sort_values(by="video_name")

    print("######### AGGREGATED COUNTS ACROSS METHODS #########")
    print(result_df)
    print()

    return result_df

count_correctness_videos(data)

# %%
def count_pred_class(data):
    for df in data:
        
        model_name = df["model_name"].unique()[0]
        
        # Count each class
        class_counts = df["pred_id"].value_counts()
        
        print(f"{'#'*10} {model_name.upper()} {'#'*10}")
        print(class_counts)
        print(f"Total: {class_counts.sum()}")
        print()
        
count_pred_class(data)