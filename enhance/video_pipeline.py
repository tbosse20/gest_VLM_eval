import os
import cv2
from tqdm import tqdm


def get_video_path(video_path: str) -> str:
    """Get the video path from command line arguments or use webcam."""
    if not video_path.endswith((".mp4", ".avi", ".mov", ".MP4")):
        raise ValueError("Video path must end with .mp4")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")

    return video_path


def setup_writer(video_path: str, cap, video_output: str = None) -> cv2.VideoWriter:
    """Setup video writer for output video."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Make sibling to video_path for output
    if not video_output:
        sibling_dir = os.path.dirname(video_path) + "_enhance"
        os.makedirs(sibling_dir, exist_ok=True)
        video_name = os.path.basename(video_path)
        video_output = os.path.join(sibling_dir, video_name)

    out = cv2.VideoWriter(video_output, fourcc, fps, (W, H))

    return out

def from_image(method, image_path: str, draw: int = 0):
    """Process a single image and return the processed image and descriptions."""
    
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found")
    if not image_path.endswith((".jpg", ".png", ".jpeg")):
        raise ValueError("Image path must end with .jpg, .png, or .jpeg")

    original_frame = cv2.imread(image_path)
    
    # Run the selected method on the frame
    frame, descriptions = method(original_frame, draw=draw)
    
    return frame, descriptions

def from_video(method, video_path: str, video_output: str = None, draw: int = 0):
    """
    Process a video file and save the output to a specified directory.
    If no output directory is specified, the video will be saved in the same directory as the input video.
    """
    
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file {video_path} not found")
    if not video_path.endswith((".mp4", ".avi", ".mov", ".MP4")):
        raise ValueError("Video path must end with .mp4, .avi, .mov, or .MP4")
    
    video_path = get_video_path(video_path) if video_path else 0
    cap = cv2.VideoCapture(video_path)
    out = setup_writer(video_path, cap, video_output) if video_path != 0 else None
    frame_count = 0

    while True:
        ret, original_frame = cap.read()
        if not ret:
            break
        original_frame = cv2.flip(original_frame, 1) if video_path == 0 else original_frame

        # Run the selected method on the frame
        frame, descriptions = method(original_frame, draw=draw)
        frame_count += 1

        # if frame_count % 8 == 0:
        #     print(f"[Frame {frame_count}] {descriptions}")
        
        if out:
            out.write(frame)
        else:
            cv2.imshow("Pose Description", frame)
            if cv2.waitKey(5) & 0xFF in [27, ord("q")]:
                break

    cap.release()
    out.release() if out else None
    cv2.destroyAllWindows()
    
    return descriptions


def from_dir(method, videos_dir: str, extension: str = "augmented", draw: int = 0):
    """
    Process all videos in a directory and save the output to a sibling directory.
    """

    # Check if the main folder exists
    if not os.path.isdir(videos_dir):
        raise FileNotFoundError(f"{videos_dir} not found")

    # Make sibling folder to videos_dir
    parent_dir = os.path.dirname(videos_dir)
    output_dir = os.path.join(parent_dir, extension)
    os.makedirs(output_dir, exist_ok=True)

    for video in tqdm(os.listdir(videos_dir), desc="Processing"):

        # Get the full path to the video file and output path
        video_path = os.path.join(videos_dir, video)
        video_output = os.path.join(output_dir, video)

        # Check if the video file exists and is a valid video file
        if not os.path.isfile(video_path):
            print(f"{video_path} not found")
            continue
        if not video_path.endswith((".mp4", ".avi", ".mov", ".MP4")):
            print(f"{video_path} is not a video file")
            continue

        # Process the video
        from_video(method, video_path, video_output, draw=draw)
        break


def main(method):
    import argparse

    parser = argparse.ArgumentParser(
        description="Pose estimation and description from video"
    )
    parser.add_argument(
        "input",
        nargs="?",  # allow 0 or 1 values
        default=None,  # default to None if omitted
        help="Path to video file or directory (default: webcam)",
    )
    parser.add_argument(
        "--draw",
        choices=[0, 1, 2],
        type=int,
        default=0,
        help="Drawing level (0: no drawing, 1: draw landmarks, 2: draw text)",
    )
    args = parser.parse_args()

    # Example usage:
    """ 
        python enhance/augment.py /path/to/input --draw 1
    """
    

    if not args.input or os.path.isfile(args.input):
        
        if args.input.endswith((".jpg", ".png", ".jpeg")):
            frame, descriptions = from_image(method, args.input, draw=args.draw)
            # Save the processed image
            file_name = os.path.basename(args.input)
            file_name_no_ext = os.path.splitext(file_name)[0]
            output_path = f"results/figures/{file_name_no_ext}_projection.jpg"
            cv2.imwrite(output_path, frame)
            print(descriptions)
        
        elif not args.input or args.input.endswith((".mp4", ".avi", ".mov", ".MP4")):
            from_video(method, args.input, draw=args.draw)

    elif os.path.isdir(args.input):
        from_dir(method, args.input, draw=args.draw)

    else:
        raise FileNotFoundError(f"{args.input} not found")


if __name__ == "__main__":
    pass
