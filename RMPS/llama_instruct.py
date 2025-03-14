import transformers
import torch
import gc
import json

def load_model():
    gc.collect()
    torch.cuda.empty_cache()

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    return pipeline

def unload(llama_package):
    pipeline = llama_package

    del pipeline

    gc.collect()
    torch.cuda.empty_cache()

def inference(messages: list, llama3_package=None):
    pipeline = llama3_package or load_model()

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

    sub_output = outputs[0]["generated_text"][-1]

    text = sub_output['content']
    start_index = text.find('{')
    end_index = text.rfind('}') + 1

    try:
        json_string = text[start_index:end_index]
        json_data = json.loads(json_string)
    except:
        print('ERROR:')
        print(json_string)
    
    unload(pipeline)

    return json_data

if __name__ == "__main__":

    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant that evaluates driving situations from a dash-cam perspective and suggests the best course of action for the ego driver."
        },
        {
            "role": "user",
            "content": """
            You are given a description of a scene from a dash-cam perspective. Your task is to evaluate the situation and suggest the best course of action for the ego driver to take. The description includes details about pedestrians, vehicles, and street conditions. Based on this information, your goal is to choose one of the following actions and provide a clear explanation of why that action is the most appropriate.

            The scene details are as follows:
            - The image depicts a narrow street with parked cars on both sides.
            - There are at least six cars visible, including a white van and a black car in the foreground.
            - A man is walking down the street, carrying a child, who is wearing a blue shirt.
            - The pedestrian in the image is standing with their arms outstretched, possibly signaling a need for assistance or indicating a specific direction.
            - The pedestrian's body language suggests that they are trying to communicate with a vehicle, possibly requesting it to stop or slow down.
            - The pedestrian is walking towards a vehicle, with their arms outstretched and legs slightly bent, indicating a clear intention to get the driver's attention.
            - The vehicle is currently traveling at a moderate speed of 0 km/h, and the pedestrian is about 10 meters (30 feet) away.

            ### Road Conditions:
            - The street is narrow, with limited space for maneuvering.
            - Parked cars on both sides could block visibility or make it difficult for the driver to pass safely.
            - It is daytime, with clear visibility, but the area is busy.

            ### Possible Actions:
            0. **Constant**
            1. **Accelerate**
            2. **Decelerate**
            3. **Brake**
            4. **Left**
            5. **Right**

            ### Instructions:
            - **Choose only one action.**
            - **Provide only one response** in the form of a JSON object with two keys: `"action"` and `"reason"`.
            - `"action"`: The selected action (one of the possible actions above).
            - `"reason"`: A brief explanation for why this action is the most appropriate.
            """
        }
    ]

    decoded_output = inference(messages)
    # print(f"Generated text: {decoded_output}")
    print(f"Action: {decoded_output['action']}")
    print(f"Reason: {decoded_output['reason']}")