import cv2

def extract_frames(video_path, start_frame=0, end_frame=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return None, None  # Return None if video cannot be opened

    # Get FPS and video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)

    # Default end_frame to last frame if None
    if end_frame is None:
        end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    # Set the start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Create video writer
    out = cv2.VideoWriter("data/sanity/output/short.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    frame_counter = start_frame

    # Loop through the video
    while cap.isOpened() and frame_counter <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to match output size
        # frame = cv2.resize(frame, size)
        out.write(frame)

        frame_counter += 1

    # Release resources
    out.release()
    cap.release()
    print(f"Extracted frames {start_frame} to {end_frame}, saved to output.mp4")

if __name__ == "__main__":
    video_path = "data/sanity/input/video_0153.mp4"
    extract_frames(video_path, start_frame=50, end_frame=55)