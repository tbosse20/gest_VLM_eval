import objects, pose

def process_frame(frame, prev_frame):
    """ Process the input frame """
    
    # Action recognition
    
    
    # Detect pedestrians
    result = pose.detect_pedestrians(frame)
    # Plot the detected poses
    frame = pose.plot_pose(result)
    
    # Detect objects
    result = objects.detect_objects(frame)
    # Plot the detected objects
    frame = objects.plot_objects(result)
    
    return frame