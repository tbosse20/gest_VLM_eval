import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import warnings
import pandas as pd
import sys
sys.path.append(".")
import config.directories as directories
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)
from sklearn.metrics import classification_report, confusion_matrix
import re
import pandas as pd
import numpy as np

classes = {
     0: "Idle",
     2: "Stop",
     3: "Advance",
     4: "Return",
     5: "Accelerate",
     6: "Decelerate",
     7: "Left",
     8: "Right",
     9: "Hail",
    10: "Attention",
    12: "Other",   
}
reverse_classes = {v: k for k, v in classes.items()}

def extract_number_and_word(text):
    """
    Extracts the number and the word following it if available.
    Handles formats like '(0) Hail', '0', '0.', '0. Stop', and 'Stop'.
    """
    if pd.isna(text):  # Handle NaN values
        return None, None

    match = re.search(
        r"(?:Answer:\s*)?"
        r"(?:\((\d+)\)\s*(\w+)?|" 
        r"\"?(\d+)\"?|" 
        r"(\d+)\.?\s*(\w+)?)",  # Only one closing parenthesis here
        str(text)
    )
    if match:
        number = match.group(1) or match.group(3) or match.group(4)
        word = match.group(2) or match.group(5)
        word = np.nan if word == "" else word
        return int(number), word

    # Try matching standalone word (e.g., "Stop") using classes dict
    match = re.search(r"^(\w+)$", str(text))
    if match and classes:
        label = match.group(1)
        number = reverse_classes.get(label, None)
        if number is None:
            raise ValueError(f"Unknown class name: '{label}'")
        return int(number), label

    return None, None
    
def post_process_caption_df(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process the caption dataframe to get the class and processed caption"""
    
    # Only keep the necessary columns
    df = df[["caption", "gt_id"]]
    
    # Extract the letter and word from the caption
    df[["pred_id", "proc_caption"]] = df["caption"].apply(
        lambda x: pd.Series(extract_number_and_word(x))
    )
    df["pred_id"] = pd.to_numeric(df["pred_id"], errors="coerce").fillna(-1)
    df["pred_id"] = df["pred_id"].astype(int)
    
    # Ensure the proc_caption and class is the same using classes
    df["confirm"] = df.apply(
        lambda x: (
            np.nan
            if pd.isna(x["proc_caption"])
            else classes.get(x["pred_id"], None) == x["proc_caption"]
        ), axis=1,
    )
    # Ensure theres no False in 'confirm'
    assert not df["confirm"].eq(False).any(), f"Error:\n{df[df['confirm'] == False]}"

    # Correct using the label
    df["correct"] = df.apply(lambda x: x["gt_id"] == x["pred_id"], axis=1)

    return df


def plot_confusion_matrix(df: pd.DataFrame, model_name: str):

    print(f"{'#'*10} {model_name.upper()} {'#'*10}")


    # Generate classification report
    print(classification_report(df["gt_id"], df["pred_id"]), "\n")

    # Get sorted list of unique labels
    labels = sorted(set(df["gt_id"]).union(set(df["pred_id"])))

    # Compute confusion matrix
    cm = confusion_matrix(df["gt_id"], df["pred_id"], labels=labels)

    # Plot confusion matrix using seaborn
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5,
        linecolor="gray",
        xticklabels=labels,
        yticklabels=labels,
    )

    # Improve plot readability
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title(
        f"Confusion Matrix Heatmap: {model_name}", fontsize=14
    )
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    # plt.show()



def post_process_csv_folder(metrics_folder):

    # Check if the folder exists
    if not os.path.exists(metrics_folder):
        raise FileNotFoundError(f"Folder '{metrics_folder}' not found.")
    if not os.path.isdir(metrics_folder):
        raise NotADirectoryError(f"'{metrics_folder}' is not a folder.")

    # Get list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(metrics_folder, "*.csv"))

    # Check if any CSV files were found
    if not csv_files:
        print("No CSV files found in the folder.")
        exit()

    # Load gt
    labels_csv = directories.LABELS_CSV
    gt = pd.read_csv(labels_csv)

    # Loop through each CSV file
    for file in csv_files:

        if "proxy" in file:
            continue  # Skip proxy files

        # Store the model name
        model_name = os.path.basename(file).split(".")[0]

        # Load the CSV file
        df = pd.read_csv(file)
        # Merge with gt on video_name and frame_idx
        # df = pd.merge(df, gt, on=["video_name", "frame_idx"])
        df = pd.merge(df, gt, on=["video_name"])

        # Post-process the caption dataframe
        df = post_process_caption_df(df)

        # Plot the confusion matrix
        plot_confusion_matrix(df, model_name)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Plot confusion matrix for captioning models."
    )
    parser.add_argument(
        "--metrics_folder",
        type=str,
        default=directories.OUTPUT_FOLDER_PATH,
        help="Path to the folder containing classification CSVs.",
    )
    args = parser.parse_args()

    # Example usage:
    """ 
    python scripts/plot_matrix.py \
        --metrics_folder results/data/captions/category
    """

    # Plot metrics
    merged_df = post_process_csv_folder(args.metrics_folder)
