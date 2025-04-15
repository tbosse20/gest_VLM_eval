# %%
# Concatenate videos in the same folder

import subprocess
import os
from tqdm import tqdm


def concat_folder_videos(parent_folder: str, include_word: str) -> None:
    """Concatenate videos from the same folder, containing the 'search word'.

    Args:
        parent_folder (str): The parent folder containing sub-folders with videos.
        include_word  (str): Only include video files containing this word.

    Input Structure:            Output Structure:
        parent_folder               parent_folder_concat
        ├── subfolder1              ├── subfolder1.mp4
        │   ├── video1.mp4          ├── subfolder2.mp4
        │   ├── video2.mp4          ...
        ├── subfolder2
        │   ├── video1.mp4
        ...
    """
    
    # Temporary file to store the list of video files (deleted automatically)
    LIST_FILE = "file_list.txt"

    # Check if the parent folder exists and is a folder
    if not os.path.exists(parent_folder):
        raise FileNotFoundError(f"Folder '{parent_folder}' not found.")
    if not os.path.isdir(parent_folder):
        raise NotADirectoryError(f"'{parent_folder}' is not a folder.")

    # Create output folder as sibling of parent folder
    output_folder = parent_folder + "_concat"
    os.makedirs(output_folder, exist_ok=True)

    # Get sub-folders
    sub_folders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]
    if len(sub_folders) == 0:
        raise FileNotFoundError(f"No sub folders found in '{parent_folder}'.")

    # Loop through sub-folders
    for sub_folder in tqdm(sub_folders, desc="Processing sub folders"):

        # Get the subfolder name as the output file
        subfolder_name = os.path.basename(sub_folder)
        output_file = os.path.join(output_folder, subfolder_name) + ".mp4"
        # Skip if the output file already exists
        if os.path.exists(output_file):
            print(f"'{subfolder_name}' already exists, skipping.")
            continue

        # Get video files containing 'search word'
        video_list = [
            f.name
            for f in os.scandir(sub_folder)
            if f.is_file() and include_word in f.name
        ]
        if len(video_list) == 0:
            print(
                f"No video files found in '{subfolder_name}' containing '{include_word}'."
            )
            continue
        # Reverse the list
        video_list = video_list[::-1]

        # Delete the file if it already exists
        if os.path.exists(LIST_FILE):
            os.remove(LIST_FILE)

        # Write video paths to the file list
        with open(LIST_FILE, "w") as f:
            for video_name in video_list:
                video_path = os.path.join(parent_folder, sub_folder, video_name)
                f.write(f"file '{video_path}'/n")

        # Run ffmpeg command
        command = [
            "ffmpeg",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            LIST_FILE,
            "-c",
            "copy",
            output_file,
        ]
        subprocess.run(command, check=True)

        # Remove the file list
        os.remove(LIST_FILE)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Concatenate videos in the same folder.")
    parser.add_argument("--parent_folder",  type=str, help="Path to the parent folder containing sub-folders.", required=True)
    parser.add_argument("--include_word",   type=str, help="Include only video files containing this word.")
    args = parser.parse_args()
    
    # Example usage
    """ 
    python concat_folder_videos.py \
        --parent_folder = "D:/realworldgestures" \
        --include_word  = "front"
    """
    
    # Concatenate videos in the same folder
    concat_folder_videos(args.parent_folder, args.include_word)