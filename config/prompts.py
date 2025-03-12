# Frame caption
frame = """
    You are driving down the road. Describe what you are seeing and what you should consider.
"""

# Pose caption
pose = """
    You are driving down the road. Examine the given image of an individual pedestrian. What are they trying to communicate to the driver using gestures.
"""

# Object caption
object = """
    You are driving down the road. Analyze the given cropped image of an object in the scene. Identify its relevance to the vehicle's decision-making process. Based on its type, position, and potential motion, determine whether it is an object the vehicle should actively monitor. Categorize the object based on the following descriptions:
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

output_prompt = """
You are an AI driving assistant. Based on the scenario provided, choose the most appropriate driving action and output your response as a single JSON object with two keys: "action" and "reason".

Possible Actions:
  0. Constant   - Maintaining speed
  1. Accelerate - Increasing speed
  2. Decelerate - Slowing down
  3. Hard Brake - Abrupt stop
  4. Left       - Changing direction left
  5. Right      - Changing direction right

Instructions:
- Choose only one action.
- Provide exactly one JSON object with the keys "action" and "reason".
- The "action" value must be one of the options above.
- The "reason" should briefly explain why this action is most appropriate for the given scenario.

GOOD EXAMPLE (correct format):
{
  "action": "Decelerate",
  "reason": "The pedestrian is indicating that we should slow down to ensure safety."
}

BAD EXAMPLES:
1) Not in JSON or missing keys:
   I think you should slow down because the pedestrian is signaling you.
2) Multiple or invalid actions:
   {
     "action": ["Decelerate", "Brake"],
     "reason": "They want the driver to stop."
   }
3) Additional text or disclaimers outside the JSON:
   Sure! Here’s my answer:
   {
     "action": "Brake",
     "reason": "Emergency situation!"
   }

Based on the scenario you receive, follow these guidelines to provide a clear and concise JSON output.
"""