import os
import pandas as pd

def stretch_sequence(sequential_csv: str, bbox_csv: str):
    """ Stretch sequences from frame-stamp to individual frames. Merge with bounding boxes. """
    
    # Check if the CSV file and video folder exist
    for csv_file in [sequential_csv, bbox_csv]:
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} does not exist.")
        if not os.path.isfile(csv_file):
            raise NotADirectoryError(f"CSV file {csv_file} is not a file.")
    
    # Load the CSV file
    seq_df = pd.read_csv(sequential_csv, index_col=False) if os.path.exists(sequential_csv) else None
    bbox_df = pd.read_csv(bbox_csv, index_col=False) if os.path.exists(bbox_csv) else None
    if seq_df.empty or bbox_df.empty:
        print("Error: One or both CSV files are empty.")
        return
    
    # Expand each row in sequence.csv into individual frames
    expanded_rows = []
    for _, row in seq_df.iterrows():
        for frame_id in range(row["start_frame"], row["end_frame"] + 1):
            expanded_rows.append({
                "video_name": row["video_name"],
                "frame_id": frame_id,
                "pedestrian_id": row["pedestrian_id"],
                "gesture_label_id": row["gesture_label_id"]
            })
    expanded_df = pd.DataFrame(expanded_rows)

    # Merge with bbox detections
    merged_df = pd.merge(
        expanded_df,
        bbox_df,
        on=["video_name", "frame_id", "pedestrian_id"],
        how="inner"  # Only keep matches
    )
    if merged_df.empty:
        print("Error: No matches found between the two CSV files.")
        return

    # Optional: sort
    merged_df = merged_df.sort_values(by=["video_name", "gesture_label_id", "frame_id", "pedestrian_id"])
    
    # Save result
    output_csv = sequential_csv.replace("sequential.csv", "stretched.csv")
    merged_df.to_csv(output_csv, index=False)
    print(f"Saved stretched annotations to: {output_csv}")

if __name__ == "__main__":
    # Example usage
    sequence_csv = "data/labels/actedgestures_sequential.csv"
    bbox_csv = "data/labels/actedgestures_bbox.csv"
    stretch_sequence(sequence_csv, bbox_csv)