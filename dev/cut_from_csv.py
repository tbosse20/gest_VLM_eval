import os
import sys
import pandas as pd
from tqdm import tqdm

sys.path.append(".")
from dev.cut_video_time import cut_video_time


def handle_csv_file(csv_file: str) -> pd.DataFrame:
    """Handle the CSV file and check if it exists.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data.
    """

    # Check if the JSON file exists
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file {csv_file} does not exist.")
    if not os.path.isfile(csv_file):
        raise NotADirectoryError(f"CSV file {csv_file} is not a file.")

    # Load the CSV file
    df = pd.read_csv(csv_file)
    # Check if the CSV file is empty
    if df.empty:
        raise ValueError(f"CSV file {csv_file} is empty.")
    # Check required columns
    required_columns = [
        "folder_path",
        "start_min",
        "start_sec",
        "end_min",
        "end_sec",
    ]
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column '{column}' in CSV file.")

    return df


def handle_main_dir(csv_file: str) -> str:

    # Get the output folder name
    videos_folder_name = os.path.basename(os.path.normpath(os.path.dirname(csv_file)))

    # Create the output folder for cut videos
    main_dir = os.path.dirname(csv_file)
    sibling_main_dir = os.path.dirname(main_dir)
    output_folder = os.path.join(sibling_main_dir, videos_folder_name + "_cut")
    os.makedirs(output_folder, exist_ok=True)

    return main_dir, output_folder


def confirm_variables(
    input_folder_path: str,
    start_time_sec: int,
    end_time_sec: int,
    input_relative_folder_path: str,
) -> bool:
    """Check if the variables are valid.

    Args:
        current_folder_path (str): Path to the current folder.
        start_time_sec (int): Start time in seconds.
        end_time_sec (int): End time in seconds.
        folder_path (str): Path to the folder.

    Returns:
        bool: True if the variables are valid, False otherwise.
    """

    # Ensure the start and end frame indices are valid
    if start_time_sec < 0 or end_time_sec < 0:
        print(f"Invalid frame indices for video {input_folder_path}. Skipping.")
        return False

    if start_time_sec >= end_time_sec:
        print(
            f"Start frame index '{start_time_sec}' is greater than or equal to last frame index '{end_time_sec}' for video '{input_relative_folder_path}'. Skipping."
        )
        return False

    if not isinstance(start_time_sec, (int)) or not isinstance(end_time_sec, (int)):
        print(
            f"Frame indices must be integers for video {input_folder_path}. Skipping."
            f"{type(start_time_sec)}, {type(end_time_sec)}"
            f"{start_time_sec}, {end_time_sec}"
        )
        return False

    # Check if the video file exists
    if not os.path.exists(input_folder_path):
        print(f"Video file {input_folder_path} does not exist. Skipping.")
        return False

    return True


def handle_video_name(
    index: int,
    current_folder_path: str,
) -> str:
    """Handle the video name and output folder.

    Args:
        index (int): Index of the video.
        current_folder_path (str): Path to the current folder.
        output_folder (str): Path to the output folder.

    Returns:
        str: Name of the video file.
    """

    is_dir = os.path.isdir(current_folder_path)

    # Set base of the output file name
    video_name = f"video_{index:02d}"

    # Convert to mp4 if not a directory
    video_name = video_name + ".mp4" if not is_dir else video_name

    return video_name


def handle_output_file_path(
    video_name: str,
    output_folder: str,
    current_folder_path: str,
) -> str:
    """Handle the output file path.

    Args:
        video_name (str): Name of the video file.
        output_folder (str): Path to the output folder.
        current_folder_path (str): Path to the current folder.

    Returns:
        str: Path to the output file.
    """

    # Check if the current folder path is a directory
    is_dir = os.path.isdir(current_folder_path)

    # Create the output file path
    output_file_path = os.path.join(output_folder, video_name)

    # Check if the output file already exists
    if os.path.exists(output_file_path):
        print(f"Output file {output_file_path} already exists. Skipping.")
        return None

    # Create the output directory if it doesn't exist
    if is_dir:
        os.makedirs(output_file_path, exist_ok=True)

    return output_file_path


def cut_with_csv(csv_file: str):
    """Cut the videos in the JSON file, using the given information.

    Args:
        csv_file (str): Path to the CSV file containing video information.

    Returns:
        None: The function saves the cut videos in the specified output folder "_cut".
    """

    # Handle the CSV file
    df = handle_csv_file(csv_file)

    # Handle the main directory and output folder
    main_dir, output_folder = handle_main_dir(csv_file)

    # Iterate through each video in the JSON file
    for index, video in tqdm(df.iterrows(), total=len(df), desc="Cutting videos"):
        input_relative_folder_path = video[
            "folder_path"
        ]  # Relative path to the video folder from the CSV file
        start_min, start_sec = video["start_min"], video["start_sec"]
        end_min, end_sec = video["end_min"], video["end_sec"]

        # Convert start and end times to seconds
        start_time_sec = start_min * 60 + start_sec
        end_time_sec = end_min * 60 + end_sec

        # Convert relative path to absolute path
        input_folder_path = os.path.join(main_dir, input_relative_folder_path)

        if not confirm_variables(
            input_folder_path,
            start_time_sec,
            end_time_sec,
            input_relative_folder_path,
        ):
            continue

        video_name = handle_video_name(
            index,
            input_folder_path,
        )
        if video_name is None:
            continue

        output_file_path = handle_output_file_path(
            video_name,
            output_folder,
            input_folder_path,
        )
        if output_file_path is None:
            continue
        
        # Cut the video using the cut_video_time function
        cut_video_time(
            input_folder_path,
            start_time_sec,
            end_time_sec,
            output_file=output_file_path,
        )

    print(f"Cut videos saved in: {output_folder}")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Cut videos using information from a CSV file."
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file containing video information.",
    )
    args = parser.parse_args()
    
    """
    folder_path,         start_min, start_sec, end_min, end_sec
            str,               int,       int,     int,     int
    videos/MVI_0059.MP4,         0,         7,       0,       9
    """

    # Example usage:
    """ 
    python dev/cut_from_csv.py \
        ../data/actedgestures2/cuts.csv                             
    """

    cut_with_csv(args.csv_file)
