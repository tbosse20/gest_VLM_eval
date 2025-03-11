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

setting_prompt = """You are an AI assistant that evaluates driving situations from a dash-cam perspective and suggests the best course of action for the ego driver. You are currently driving 10 km/h."""

task_prompt = """You are given a description of a scene from a dash-cam perspective. Your task is to evaluate the situation and suggest the best course of action for the ego driver to take. The description includes details about pedestrians, vehicles, and street conditions. Based on this information, your goal is to choose one of the following actions and provide a clear explanation of why that action is the most appropriate.\n\n
The scene details are as follows:\n"""

output_prompt = prompt = """
You are an AI driving assistant. Based on the scenario provided, choose the most appropriate driving action and output your response as a single JSON object with two keys: "action" and "reason".

Possible Actions:
  0. Constant speed  # Maintaining speed
  1. Accelerate      # Increasing speed
  2. Decelerate      # Slowing down
  3. Hard Brake      # Abrupt stop
  4. Turn left       # Changing direction left
  5. Turn right      # Changing direction right

Instructions:
- Choose only one action.
- Provide exactly one JSON object with the keys "action" and "reason".
- The "action" value must be one of the options above.
- The "reason" should briefly explain why this action is most appropriate for the given scenario.

Positive Example:
Scenario: The vehicle is approaching a red light.
Expected Output: {"action": "Decelerate", "reason": "Reducing speed to stop safely at the red light."}

Negative Example:
Scenario: The vehicle is stopped in a safe zone.
Incorrect Output: {"action": "Accelerate", "reason": "Increasing speed from a standstill is unnecessary in a safe zone."}
(Explanation: Accelerating is not appropriate because the vehicle is already stationary in a safe area.)

Based on the scenario you receive, follow these guidelines to provide a clear and concise JSON output.
"""