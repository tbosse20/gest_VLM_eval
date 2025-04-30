import sys
sys.path.append(".")
import config.flags as flags

prompts = {
    # "blank": "",
    # "determine": "Determine what gesture the pedestrian is making.",
    # "body": "Provide a detailed explanation of the pedestrian’s body posture and movements.",
    # "context": "You are an autonomous vehicle navigating a road. Determine what gesture the pedestrian is making.",
    # "objective": "You are an autonomous vehicle navigating a road. Determine what gesture the pedestrian is making. Your response will be used by an AI system to make real-time driving decisions.",
    
    "category": """
    Analyze the pedestrian's gesture in the given image and determine the action they are requesting from the ego driver.  
    Follow these steps:  
    1. Identify the pedestrian's gesture (e.g., waving hand, pointing, holding out palm).  
    2. Consider the context (e.g., crosswalk, roadside, intersection).  
    3. Select the most appropriate action the ego driver should take based on the gesture.  

    Use the following format for your response:  
    "<selected_option>"

    **Options:**  
     0. Idle       - No movement or gestures.  
     2. Stop       - Stopping in any manner.  
     3. Advance    - Drive forward in any manner.  
     4. Return     - Backup by reverse or turn the vehicle.  
     5. Accelerate - Increase current speed.  
     6. Decelerate - Decrease current speed.  
     7. Left       - Turn to the left lane.  
     8. Right      - Turn to the right lane.  
     9. Hail       - Hail for a ride.  
    10. Point      - Pointing in any manner.  
    11. Attention  - Seeking awareness.  
    12. Other      - Non‑navigation gesture.  
    13. Unclear    - Unknown or unclear.  

    Ensure that your response includes only **one** category in the specified format.
    """ + 
    "The pose is projected upon the person, to help understand their pose." if flags.projection_enhancement else "",
    
}