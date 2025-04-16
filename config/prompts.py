prompts = {
    "blank": "",
    "determine": "Determine what gesture the pedestrian is making.",
    "body": "Provide a detailed explanation of the pedestrian’s body posture and movements.",
    "context": "You are an autonomous vehicle navigating a road. Determine what gesture the pedestrian is making.",
    "objective": "You are an autonomous vehicle navigating a road. Determine what gesture the pedestrian is making. Your response will be used by an AI system to make real-time driving decisions.",
    
    "category": """
    Analyze the pedestrian's gesture in the given image and determine the action they are requesting from the ego driver.  
    Follow these steps:  
    1. Identify the pedestrian's gesture (e.g., waving hand, pointing, holding out palm).  
    2. Consider the context (e.g., crosswalk, roadside, intersection).  
    3. Select the most appropriate action the ego driver should take based on the gesture.  

    Use the following format for your response:  
    "<selected_option>"

    **Options:**  
    a. Follow - The pedestrian is signaling the driver to follow them.  
    b. Hail - The pedestrian is trying to get the driver’s attention (e.g., waving).  
    c. Forward - The pedestrian is motioning the driver to move forward.  
    d. Left - The pedestrian is signaling the driver to turn left.  
    e. Idle - The pedestrian is not making any clear gesture; the driver should wait.  
    f. Reverse - The pedestrian is requesting the driver to move backward.  
    g. Stop - The pedestrian is signaling the driver to stop (e.g., raised hand, palm facing forward).  
    h. Other - The gesture does not match any of the above categories.
    i. Right - The pedestrian is signaling the driver to turn right.    

    Ensure that your response includes only **one** category in the specified format.
    """
}