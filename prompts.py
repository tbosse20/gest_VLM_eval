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