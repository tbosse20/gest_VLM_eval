# Cut a video file by start time to end time or duration

import subprocess
import os


def get_fps(video_file: str) -> float:
    """Get the frames per second (fps) of a video file using ffprobe.

    Args:
        video_file (str): Path to the video file.

    Returns:
        float: Frames per second of the video.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_file,
    ]
    output = subprocess.check_output(cmd).strip()
    try:
        num, denom = map(int, output.decode("utf-8").split("/"))
        return num / denom
    except ValueError:
        raise RuntimeError(f"Could not parse fps from ffprobe output: {output}")


def cut_video_time(
    input_file: str, start_time: int, end_time: int = None, duration: int = None, output_file: str = None
) -> None:
    """Cut a video file using ffmpeg with start time and end time or duration.

    Args:
        input_file  (str): Path to the input video file.
        start_time  (int): Start time in seconds.
        end_time    (int): End time in seconds.
        duration    (int): Duration in seconds.

    Returns:
        None: A new video file is created in a sibling folder of the input file.
    """
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file '{input_file}' not found.")
    
    # Ensure time parameters are positive integers
    if start_time < 0 and not isinstance(start_time, int):
        raise ValueError("Start time must be a positive integer.")
    if end_time is not None and end_time < 0 and not isinstance(end_time, int):
        raise ValueError("End time must be a positive integer.")
    if duration is not None and duration <= 0 and not isinstance(duration, int):
        raise ValueError("Duration must be a positive integer.")
    
    # Check if either end_time or duration is provided (but not both)
    if not end_time and not duration:
        raise ValueError("Either 'end_time' or 'duration' must be provided.")
    if end_time and duration:
        raise ValueError("Only one of 'end_time' or 'duration' can be provided.")

    # Make output folder as sibling of input files parent folder
    if not output_file:
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
        "-loglevel", "error",   # Suppress unnecessary libx264 info
        "-ss", str(start_time),
        "-i", input_file,
        "-t", str(duration),
        "-an",                  # Remove audio
        "-c:v", "libx264",      # Re-encode video
        "-preset", "fast",      # Speed vs compression tradeoff
        output_file,
    ]
    subprocess.run(command, check=True)

    # Print output file path
    fps = get_fps(input_file)
    if end_time is not None:
        print(f"Duration: {duration} seconds")
        print(f"Frames: {int(duration * fps)} frames")  # Assuming 30 fps
    else:
        print(f"End time: {end_time} seconds")
        print(f"End frame: {int(end_time * fps)} frames")  # Assuming 30 fps


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Cut video by time.")
    parser.add_argument("--input_file", type=str, help="Path to the input video file.", required=True)
    parser.add_argument("--start_time", type=int, help="Start time in seconds.",        required=True)
    parser.add_argument("--end_time",   type=int, help="End time in seconds.")
    parser.add_argument("--duration",   type=int, help="Duration in seconds.")
    args = parser.parse_args()

    # Example usage
    """
    python cut_video.py \
        --input_file  = "../data/actedgestures_original/video_01.MP4" \
        --start_time  = 1 * 60 + 30 \
        --end_time    = 1 * 60 + 40
    """

    cut_video_time(
        input_file=args.input_file,
        start_time=args.start_time,
        end_time=args.end_time,
        duration=args.duration,
    )