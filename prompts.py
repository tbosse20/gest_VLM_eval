# Pose caption
pose = """
    Examine the given image of an individual pedestrian. Analyze their body posture, limb positioning, and gestures to infer their intended communication towards a vehicle. Based on their stance and movement cues, select the most accurate interpretation of their intent:
"""

# Object caption
object = """
    Analyze the given cropped image of an object in the scene. Identify its relevance to the vehicle's decision-making process. Based on its type, position, and potential motion, determine whether it is an object the vehicle should actively monitor. Categorize the object based on the following descriptions:
"""

# Frame caption
frame = """
    Examine the given full-frame image representing the vehicle's surroundings. Identify all relevant objects, their spatial relationships, and their impact on the vehicle's decision-making. Based on the scene composition, categorize the overall driving context using the most appropriate description:
"""
# frame += """
#     For example, the scene may indicate a **clear path** if the road ahead is free of obstacles, moving objects, or immediate hazards. If a pedestrian is present and may require the vehicle to stop, yield, or adjust speed, the scene falls under **pedestrian interaction**. When other vehicles in proximity influence movement, such as merging, stopping, or turning, the scene involves **vehicle interaction**.

#     If traffic signals, signs, or road markings are visible and should be considered, describe the scene as a **traffic-controlled area**. A stationary object or road blockage that requires the vehicle to maneuver or stop would indicate an **obstacle ahead**. When multiple moving objects—such as pedestrians, cyclists, or vehicles—require the vehicle to anticipate interactions, the scene can be described as **dynamic**.

#     In cases where unexpected or dangerous elements, such as accidents, sudden pedestrian crossings, or emergency vehicle presence, impact decision-making, classify the scene as an **emergency or hazardous situation**. If the scene lacks enough information for a definitive classification, describe it as **unclear**.

#     Provide a concise description that best fits the scene based on these examples.
# """

init_prompt = """
    You are given a description of a scene from a dash-cam perspective. Your task is to evaluate the situation and suggest the best course of action for the ego driver to take. The description includes details about pedestrians, vehicles, and street conditions. Based on this information, your goal is to choose one of the following actions and provide a clear explanation of why that action is the most appropriate.

    The scene details are as follows:
"""

task_prompt = """
### Possible Actions:
1. Accelerate
2. Decelerate
3. Brake
4. Turn left
5. Turn right

### Task:
- Choose one action from the list above and respond **only with the number and action**, in this format: <number>. <action>
- Do **not** add any extra text, markdown, or explanations.

### Positive Examples (Correct Responses):
- "2. Decelerate"
- "3. Brake"
- "1. Accelerate"
- "4. Turn left"
- "5. Turn right"

### Negative Examples (Incorrect Responses):
- "3. Brake ###"  # Incorrect, extra characters
- "### 3. Brake"  # Incorrect, extra markdown
- "The correct action is 3. Brake."  # Incorrect, extra text
- "I would suggest 3. Brake."  # Incorrect, explanation
- "Action 3: Brake"  # Incorrect, wrong format

Action: 
"""