# %%
# Concatenate videos in the same folder

import subprocess
import os
from tqdm import tqdm

def concat_folder_videos(parent_folder: str, include_word: str) -> None:
    """ Concatenate videos from the same folder, containing the 'search word'.
    
    Args:
        parent_folder (str): The parent folder containing subfolders with videos.
        include_word  (str): Only include video files containing this word.
    
    Structure:
        parent_folder
        ├── subfolder1
        │   ├── video1.mp4
        │   ├── video2.mp4
        ├── subfolder2
        │   ├── video1.mp4
        ...
    
    Output:
        parent_folder_concat
        ├── subfolder1.mp4
        ├── subfolder2.mp4
        ...
    """
    # File to store the list of video files (deleted automatically)
    LIST_FILE = "file_list.txt"
    
    # Check if the parent folder exists and is a folder    
    if not os.path.exists(parent_folder):
        raise FileNotFoundError(f"Folder '{parent_folder}' not found.")
    if not os.path.isdir(parent_folder):
        raise NotADirectoryError(f"'{parent_folder}' is not a folder.")
    
    # Create output folder as sibling of parent folder
    output_folder = parent_folder + "_concat"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get subfolders
    subfolders = [
        f.path
        for f in os.scandir(parent_folder)
        if f.is_dir()
    ]
    if len(subfolders) == 0:
        raise FileNotFoundError(f"No subfolders found in '{parent_folder}'.")
    
    # Loop through subfolders
    for sub_folder in tqdm(subfolders, desc="Processing subfolders"):
        
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
            print(f"No video files found in '{subfolder_name}' containing '{include_word}'.")
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
            "ffmpeg", "-f", "concat", "-safe", "0",
            "-i", LIST_FILE, "-c", "copy", output_file
        ]
        subprocess.run(command, check=True)
        
        # Remove the file list
        os.remove(LIST_FILE)

concat_folder_videos(
    parent_folder = "D:/realworldgestures",
    include_word  = "front",
)

# %%
import subprocess
import os

def concat_folder_videos(input_folder, video_list, output_folder) -> None:
    # Create a file list for ffmpeg
    list_file = "file_list.txt"
    with open(list_file, "w") as f:
        for video in video_list:
            f.write(f"file '{input_folder}/{video}'/n")

    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.basename(video_list[0]) + "_concat.mp4"
    output_file = os.path.join(output_folder, output_file)
    
    # Run ffmpeg command
    command = [
        "ffmpeg", "-f", "concat", "-safe", "0",
        "-i", list_file, "-c", "copy", output_file
    ]
    subprocess.run(command, check=True)
    
    # Remove the file list
    os.remove(list_file)
concat_folder_videos(
    input_folder  = "D:/realworldgestures_concat_cut",
    output_folder = "D:/realworldgestures_concat_cut",
    video_list    = ["2025-03-15_13-15-15 (6).mp4", "2025-03-15_13-23-36 (6).mp4"],
)

# %%
# Cut a video file by start time to end time or duration

import subprocess
import os

def cut_video(input_file: str, start_time: int, end_time: int = None, duration: int = None) -> None:
    """ Cut a video file using ffmpeg. 
    
    Args:
        input_file  (str): Path to the input video file.
        start_time  (int): Start time in seconds.
        end_time    (int): End time in seconds.
        duration    (int): Duration in seconds.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    # Check if either end_time or duration is provided (but not both)
    if not end_time and not duration:
        raise ValueError("Either 'end_time' or 'duration' must be provided.")
    if end_time and duration:
        raise ValueError("Only one of 'end_time' or 'duration' can be provided.")
    
    # Make output folder as sibling of input files parent folder
    output_folder = os.path.dirname(input_file) + "_cut"
    output_file = os.path.join(output_folder, os.path.basename(input_file))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Check if output file already exists
    if os.path.exists(output_file):
        raise FileExistsError(f"Output file '{output_file}' already exists.")
    
    # Calculate duration if end_time is provided
    duration = end_time - start_time if end_time else duration
    
    # Run ffmpeg command
    command = [
        "ffmpeg",
        "-i",  input_file,      # Input file
        "-ss", str(start_time), # Start time (in seconds)
        "-t",  str(duration),   # Duration (in seconds)
        "-c",  "copy",          # Copy streams without re-encoding
        output_file             # Output file
    ]
    subprocess.run(command, check=True)
cut_video(
    input_file  = "C:/Users/Tonko/OneDrive/Dokumenter/School/Merced/reconstruction/MVI_0050.MP4",
    start_time  = 1 * 60 + 30, # Min * 60 + Sec
    end_time    = 1 * 60 + 40, # Min * 60 + Sec
)


# %% 

import subprocess
import argparse

def get_duration(video_file):
    """
    Retrieve the duration of a video file in seconds using ffprobe.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_file
    ]
    output = subprocess.check_output(cmd).strip()
    try:
        return float(output)
    except ValueError:
        raise RuntimeError(f"Could not parse duration from ffprobe output: {output}")

def build_atempo_chain(speed_factor):
    """
    Build a chain of atempo filters such that each factor is within [0.5, 2.0].
    For example, a speed_factor of 0.4065 is broken into atempo=0.5,atempo=0.813 (0.5 * 0.813 ≈ 0.4065).
    """
    factors = []
    # If the speed factor is greater than 2.0, repeatedly apply atempo=2.0.
    while speed_factor > 2.0:
        factors.append(2.0)
        speed_factor /= 2.0
    # If the speed factor is less than 0.5, repeatedly apply atempo=0.5.
    while speed_factor < 0.5:
        factors.append(0.5)
        speed_factor /= 0.5
    # Append the remaining factor (if it's 1.0, it doesn't change the audio).
    factors.append(speed_factor)
    
    # Build the filter chain string.
    atempo_chain = ",".join([f"atempo={f:.6f}" for f in factors])
    return atempo_chain

def adjust_video_speed(edit_video, fixed_video):
    """
    Adjust the speed of the edit_video to match the duration of the fixed_video.
    """
    
    if not os.path.exists(edit_video):
        raise FileNotFoundError(f"Edit video file not found: {edit_video}")
    if not os.path.exists(fixed_video):
        raise FileNotFoundError(f"Fixed video file not found: {fixed_video}")
    
    # Get durations of both videos.
    edit_duration = get_duration(edit_video)
    fixed_duration = get_duration(fixed_video)
    
    # Calculate the speed factor.
    # A speed_factor < 1 means the edit video must be sped up (and vice versa).
    speed_factor = edit_duration / fixed_duration
    # For video, setpts uses the reciprocal factor.
    pts_factor = 1.0 / speed_factor
    
    # Build the appropriate chain for atempo so that each factor is within the allowed range.
    atempo_chain = build_atempo_chain(speed_factor)
    print(f"Using atempo chain for audio: {atempo_chain}")
    
    # Construct the filter_complex string.
    filter_complex = f"[0:v]setpts={pts_factor:.6f}*PTS[v];[0:a]{atempo_chain}[a]"
    
    # Create an output folder for the sped-up video.
    output_folder = os.path.dirname(edit_video) + "_sped"
    output_file = os.path.join(output_folder, os.path.basename(edit_video))
    os.makedirs(output_folder, exist_ok=True)
    
    # Remove the output file if it already exists.    
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Build and run the ffmpeg command.
    cmd = [
        'ffmpeg', '-i', edit_video,
        '-filter_complex', filter_complex,
        '-map', '[v]', '-map', '[a]',
        output_file
    ]
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    output_duration = get_duration(output_file)
    if abs(output_duration - fixed_duration) > 1.0:
        os.remove(output_file)
        raise RuntimeError(f"Output video duration {output_duration}s differs from target duration {fixed_duration}s")
    
adjust_video_speed(
    edit_video   = "C:/Users/Tonko/OneDrive/Dokumenter/School/Merced/reconstruction_cut/MVI_0050.MP4",
    fixed_video  = "C:/Users/Tonko/OneDrive/Dokumenter/School/Merced/actedgestures/Reverse.MP4",
)