import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(".")
from scripts.src.compute_similarity_metrics import compute_similarity_metrics


def make_sibling_folder(folder_path: str, sibling_name: str):
    """Create a sibling folder to the input folder."""

    # Get the parent directory (where sibling folders exist)
    parent_dir = Path(folder_path).parent

    # Define the path for the new sibling folder
    sibling_folder = parent_dir / sibling_name

    # Create the sibling folder if it doesn't exist
    sibling_folder.mkdir(parents=True, exist_ok=True)

    return sibling_folder


def process_csv(label_caption_csv, gen_caption_folder):
    """Processes a CSV file, generates captions, computes similarity metrics, and saves results."""

    if not os.path.exists(label_caption_csv):
        raise FileNotFoundError(f"Input CSV file '{label_caption_csv}' not found.")

    COLUMNS = ["video_name", "frame_idx", "prompt_type"]
    METRICS = ["cosine"]  # , "jaccard", "bleu", "meteor", "rouge_l", "bert"]
    METRICS.sort()

    # Load CSV
    label_df = pd.read_csv(label_caption_csv)

    # Ensure necessary columns exist
    required_columns = {"video_name", "frame_idx", "caption"}
    if not required_columns.issubset(label_df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")

    # Get list of generated caption CSVs (ex.: results/data/CAPTIONS/proxy.csv)
    gen_caption_csvs = [
        os.path.join(gen_caption_folder, f)
        for f in os.listdir(gen_caption_folder)
        if f.endswith(".csv")
    ]

    # Make sibling folder to gen_caption_folder
    metrics_folder = make_sibling_folder(gen_caption_folder, "metrics")

    # Loop through generated caption CSVs
    for gen_caption_csv in gen_caption_csvs:

        # Get metric csv file (ex.: results/data/METRICS/proxy.csv)
        metric_path = os.path.join(metrics_folder, os.path.basename(gen_caption_csv))

        # Make or skip metric csv file
        if not os.path.exists(metric_path):
            pd.DataFrame(columns=COLUMNS + METRICS).to_csv(
                metric_path, index=False, header=True
            )

        # Load generated captions
        gen_df = pd.read_csv(gen_caption_csv)

        # Process each row
        gen_caption_csv_name = os.path.basename(gen_caption_csv).split(".")[0]
        for index, row in tqdm(
            gen_df.iterrows(),
            total=gen_df.shape[0],
            desc=f"Proc. {gen_caption_csv_name}",
        ):
            # Get image name and frame index
            video_name, frame_idx = row["video_name"], row["frame_idx"]

            # Retrieve the corresponding ground truth caption
            label_caption = label_df.loc[
                (
                    (label_df["video_name"] == video_name)
                    & (label_df["frame_idx"] == frame_idx)
                ),
                "caption",
            ]
            if label_caption.empty or label_caption.values[0] in [None, "empty"]:
                continue
            label_caption = label_caption.values[0]

            # Get the predicted caption
            pred_caption = row["caption"]
            if pred_caption in [None, "empty"]:
                continue

            # Compute similarity metrics
            metrics_df = compute_similarity_metrics(label_caption, pred_caption)

            # Get prompt type if available
            prompt_type = row["prompt_type"] if "prompt_type" in row else None
            # Save results to CSV
            frame_sample_df = pd.DataFrame(
                {
                    "video_name": [video_name],
                    "frame_idx": [frame_idx],
                    "prompt_type": [prompt_type],
                }
            )
            frame_sample_df = pd.concat([frame_sample_df, metrics_df], axis=1)
            frame_sample_df.to_csv(metric_path, mode="a", index=False, header=False)


# Example Usage
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Compare generated captions with human captions."
    )
    parser.add_argument(
        "--label_caption_csv",
        type=str,
        default="../data/actedgestures/label.csv",
        help="Path to the CSV file containing ground truth captions.",
    )
    parser.add_argument(
        "--gen_caption_folder",
        type=str,
        default="results/data/captions/",
        help="Folder containing generated captions CSV files.",
    )
    args = parser.parse_args()

    # Example usage:
    """ 
    python scripts/compare_captions.py \
        --label_caption_csv  ../data/actedgestures/label.csv" \
        --gen_caption_folder results/data/captions/
    """

    process_csv(
        label_caption_csv=args.label_caption_csv,
        gen_caption_folder=args.gen_caption_folder,
    )
