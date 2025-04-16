import cv2
import os
from tqdm import tqdm


def save_video_frames(video_path, output_folder, interval: int = 1) -> None:
    """ 
    Convert video to frames to specific folder.
    
    Args:
        video_path (str):    Path to the video file.
        output_folder (str): Path to the folder where frames will be saved.
        interval (int):      Interval for saving frames (fps / interval => new_fps).
    
    Returns:
        None: Saves frames to the specified folder.
    """

    # Create output folder if it doesn't exist
    if os.path.exists(output_folder):
        print(f"'{output_folder}' already exists, skipping.")
        return
    else:
        os.makedirs(output_folder, exist_ok=True)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get total number of frames
    frame_idx = 0
    # Get video name
    video_name = output_folder.split("/")[-1]

    # Loop through each frame
    with tqdm(total=frame_count, desc=f"Proc. {video_name}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame in interval
            if frame_idx % interval == 0:
                frame_filename = os.path.join(
                    output_folder, f"frame_{frame_idx:04d}.png"
                )
                cv2.imwrite(frame_filename, frame)

            # Update progress bar
            pbar.update(1)
            frame_idx += 1

    cap.release()


def save_video_frames_folder(videos_folder: str, interval: int) -> None:
    """
    Save frames from all videos in a new folder with "_frames" suffix.

    Args:
        videos_folder (str): Path to the folder containing videos.
        interval (int):      Interval for saving frames (fps / interval => new_fps).

    Input Structure:        Output Structure:
        folder/                 folder_frames/
        ├── video1.mp4          ├── video1/
        │                       │   ├── frame_0001.png
        │                       │   ├── frame_0002.png
        ├── video2.mp4          ├── video2/
        │                       │   ├── frame_0001.png
        │                       │   ├── frame_0002.png
        ...                       ...

    Returns:
        None: Saves frames to the sibling folder with "_frames" suffix.
    """

    # Validate input folder
    if not os.path.exists(videos_folder):
        raise FileNotFoundError(f"Input folder '{videos_folder}' not found.")
    if not os.path.isdir(videos_folder):
        raise NotADirectoryError(f"Input folder '{videos_folder}' is not a directory")

    # Create output folder if it doesn't exist
    output_folder = videos_folder + "_frames"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Loop through all video names in the folder
    video_names = os.listdir(videos_folder)
    for video_name in video_names:
        # Skip non-video files
        if not video_name.endswith(".mp4"):
            continue

        # Get video path
        video_path = f"{videos_folder}/{video_name}"

        # Create output folder for video frames
        video_output_folder = f"{output_folder}/{video_name.split('.')[0]}"

        # Create output folder if it doesn't exist
        save_video_frames(video_path, video_output_folder, interval)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from videos")
    parser.add_argument(
        "--videos_folder",
        type=str,
        default="../realworldgestures",
        help="Folder containing videos",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
        help="Interval for saving frames (fps / interval => new_fps)",
    )
    args = parser.parse_args()

    # Example usage
    """ 
    python video_to_frames.py \
        --videos_folder ../realworldgestures \
        --interval 18
    """

    save_video_frames_folder(args.videos_folder, args.interval)
