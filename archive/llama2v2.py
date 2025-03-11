import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def load_model():
    gc.collect()
    torch.cuda.empty_cache()

    # model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    print('Loading model: Llama-3-8B')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically assigns to GPU
        # torch_dtype=torch.float16,
        # load_in_4bit=True  # Directly use 4-bit loading
    )

    model.eval()
    return model, tokenizer, device


def unload(llama_package):
    model, tokenizer, device = llama_package

    del model
    del tokenizer
    del device

    gc.collect()
    torch.cuda.empty_cache()

def inference(prompt: str, llama3_package=None):
    model, tokenizer, device = llama3_package or load_model()

    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,  # Reduce to prevent memory overload
                temperature=0.7,
                do_sample=True  # Enable sampling for better diversity
            )

            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_output

    except torch.cuda.OutOfMemoryError:
        print("⚠️ Out of Memory! Try reducing max_new_tokens or using CPU mode.")
        torch.cuda.empty_cache()
        return None

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

    

if __name__ == "__main__":

    init_prompt = """
        You are given a description of a scene from a dash-cam perspective. Your task is to evaluate the situation and suggest the best course of action for the ego driver to take. The description includes details about pedestrians, vehicles, and street conditions. Based on this information, your goal is to choose one of the following actions and provide a clear explanation of why that action is the most appropriate.

        ### Scene Description:
    """

    scene_prompt = """
        A person is standing in front of you.    
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

    prompt = "\n".join([init_prompt, scene_prompt, task_prompt])

    decoded_output = inference(prompt)

    print(f"Generated text: {decoded_output}")
