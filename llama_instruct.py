import transformers
import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are an AI assistant that evaluates driving situations from a dash-cam perspective and suggests the best course of action for the ego driver."},
    {"role": "user", "content": "You are given a description of a scene from a dash-cam perspective. Your task is to evaluate the situation and suggest the best course of action for the ego driver to take. The description includes details about pedestrians, vehicles, and street conditions. Based on this information, your goal is to choose one of the following actions and provide a clear explanation of why that action is the most appropriate.\n\nThe scene details are as follows:\nThe image depicts a narrow street with parked cars on both sides. There are at least six cars visible, including a white van and a black car in the foreground. A man is walking down the street, carrying a child, who is wearing a blue shirt. The scene suggests that the vehicle is navigating through this busy street, possibly looking for a parking spot or driving to a destination. The presence of pedestrians and parked cars indicates that the driver needs to be cautious and attentive to avoid any accidents. The pedestrian in the image is standing with their arms outstretched, possibly signaling a need for assistance or indicating a specific direction. Their body language suggests that they are trying to communicate with a vehicle, perhaps requesting it to stop or slow down. The presence of lines connecting different parts of their body further emphasizes the importance of their gestures and adds a visual element to their communication. The pedestrian in the image is walking towards a vehicle, with their arms outstretched and legs slightly bent. Their body posture suggests that they are signaling to the driver of the vehicle to stop or slow down. The pedestrian's outstretched arms and forward-leaning stance indicate a clear intention to communicate with the driver. Based on these cues, it can be inferred that the pedestrian is attempting to get the driver's attention and convey a message related to traffic safety.\n\nPossible Actions:\n\n1. Accelerate  \n2. Decelerate  \n3. Brake  \n4. Turn left  \n5. Turn right  \n\n### Instructions:\n- **Choose only one action.**  \n- **Provide only one response** in the form of a JSON object with two keys: `\"action\"` and `\"reason\"`.  \n- `\"action\"`: The selected action (one of the possible actions above).  \n- `\"reason\"`: A short explanation for why this action is the most appropriate."},
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)

print(outputs[0]["generated_text"][-1])
