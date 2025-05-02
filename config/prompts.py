import sys
sys.path.append(".")
import config.flags as flags

category_prompt = """
    Analyze the pedestrian's gesture in the given image and determine the action they are requesting from the ego driver.  

    **Options:**  
     0. Idle       - No gestures.
     2. Stop       - Stopping in any manner.
     3. Advance    - Drive forward in any manner.
     4. Return     - Backup by reverse or turn the vehicle.
     5. Accelerate - Increase current speed.
     6. Decelerate - Decrease current speed.
     7. Left       - Turn to the left lane.
     8. Right      - Turn to the right lane.
     9. Hail       - Hail for a ride.
    10. Attention  - Seeking awareness.
    12. Other      - Irrelevant gesture.
"""

# Append enhancement note only if the flag is enabled
if flags.projection_enhancement:
    category_prompt += "The pose is projected upon the person, to help understand their pose."

prompts = {
    # "blank": "",
    # "determine": "Determine what gesture the pedestrian is making.",
    # "body": "Provide a detailed explanation of the pedestrian’s body posture and movements.",
    # "context": "You are an autonomous vehicle navigating a road. Determine what gesture the pedestrian is making.",
    # "objective": "You are an autonomous vehicle navigating a road. Determine what gesture the pedestrian is making. Your response will be used by an AI system to make real-time driving decisions.",
    "category": category_prompt,
}