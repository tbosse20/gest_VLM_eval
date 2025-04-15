import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import warnings
import pandas as pd

warnings.simplefilter(action="ignore", category=pd.errors.SettingWithCopyWarning)

classes = {
    "a": "follow",
    "b": "hail",
    "c": "forward",
    "d": "left",
    "e": "idle",
    "f": "reverse",
    "g": "stop",
    "h": "other",
    "i": "right",
}


def extract_letter_and_word(text):
    """
    Extracts the letter and the word following it if available.
    Handles formats like '(b) Hail', 'b', 'e.', and 'e. Stop'.
    """
    if pd.isna(text):  # Handle NaN values
        return None, None

    # Updated regex pattern to capture all variations, including missing "Answer:"
    match = re.search(
        r"(?:Answer:\s*)?(?:\((\w)\)\s*(\w+)?)|"  # (A) Apple
        r'(?:Answer:\s*)?"([a-zA-Z])"|'  # "a" or "A"
        r"(?:Answer:\s*)?([a-zA-Z])|"  # a or A
        r"(?:Answer:\s*)?([a-zA-Z])\.(\s*\w+)?",  # A. Apple
        text,
    )

    if match:
        letter = (
            match.group(1) or match.group(3) or match.group(4)
        )  # Extract letter from different formats
        word = (
            match.group(2)
            if match.group(2)
            else (match.group(5).strip() if match.group(5) else None)
        )
        word = np.nan if word == "" else word
        return letter, word.lower() if word else None
    return None, None  # Return None if no match is found


def post_process_caption_df(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process the caption dataframe to get the class and processed caption"""

    # Only keep the necessary columns
    df = df[["caption", "label"]]

    # Extract the letter and word from the caption
    df[["class", "proc_caption"]] = df["caption"].apply(
        lambda x: pd.Series(extract_letter_and_word(x))
    )

    # Ensure the proc_caption and class is the same using classes
    df["confirm"] = df.apply(
        lambda x: (
            np.nan
            if pd.isna(x["proc_caption"])
            else classes.get(x["class"], None) == x["proc_caption"]
        ),
        axis=1,
    )
    # Ensure theres no False in 'confirm'
    assert not df["confirm"].eq(False).any(), f"Error:\n{df[df['confirm'] == False]}"

    # Set the processed caption to the class using the letter
    df["proc_caption"] = df.apply(
        lambda x: (classes.get(x["class"], np.nan).lower()), axis=1
    )

    # Correct using the label
    df["correct"] = df.apply(lambda x: x["label"].lower() == x["proc_caption"], axis=1)

    return df


def plot_confusion_matrix(df: pd.DataFrame, model_name: str):

    # Compute the accuracy
    accuracy = df["correct"].mean()
    print(f"{model_name} accuracy: {accuracy:.2f}")

    from sklearn.metrics import classification_report, confusion_matrix

    print(df[["label", "proc_caption"]])

    # Generate classification report
    print(classification_report(df["label"], df["proc_caption"]))

    # Get sorted list of unique labels
    labels = sorted(set(df["label"]).union(set(df["proc_caption"])))

    # Compute confusion matrix
    cm = confusion_matrix(df["label"], df["proc_caption"], labels=labels)

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
        f"Confusion Matrix Heatmap: {model_name} (Acc = {accuracy:.2f})", fontsize=14
    )
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


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
    gt = pd.read_csv("data/labels/firsthand_category.csv")

    # Loop through each CSV file
    for file in csv_files:

        if "proxy" in file:
            continue  # Skip proxy files

        # Store the model name
        model_name = os.path.basename(file).split(".")[0]

        # Load the CSV file
        df = pd.read_csv(file)
        # Merge with gt on video_name and frame_idx
        df = pd.merge(df, gt, on=["video_name", "frame_idx"])

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
        default="results/data/captions/category",
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
