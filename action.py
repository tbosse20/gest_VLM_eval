import cv2

def recognize_action(frame, prev_frame):
    """ Recognize the action from the input frames. """
    
    if prev_frame is None: return None
    
    prompt = """ What driving action is the ego vehicle performing? """
    
    # Predict action from frames
    results = SOME_COOL_VLM(frame, prev_frame, prompt)
    
    return results

if __name__ == "__main__":
    # Run video and action recognition
    video_path = 'data/sanity/video_0153.mp4'
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        action = recognize_action(frame, prev_frame)
        print(action)
        prev_frame = frame.copy()
    cap.release()
    