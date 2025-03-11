import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def load_model():
    gc.collect()
    torch.cuda.empty_cache()

    print('Loading model: Llama-3-8B')

    model_name = "meta-llama/Meta-Llama-3-8B"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically assigns to GPU
        torch_dtype=torch.float16,
        load_in_4bit=True  # Directly use 4-bit loading
    )

    model.eval()
    return model, tokenizer, device


def unload(llama2_package):
    model, tokenizer, device = llama2_package

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
    decoded_output = inference("The color of the sun is")
    print(f"Generated text: {decoded_output}")
