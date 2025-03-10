import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model():
    torch.cuda.empty_cache()

    model_name = "meta-llama/Meta-Llama-3-8B"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_compute_dtype = torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
    ).to(device)

    return model, tokenizer, device

def unload_model(model, tokenizer, device):
    del model
    model = None
    del tokenizer
    tokenizer = None
    del device
    device = None

def inference(prompt: str):

    model, tokenizer, device = load_model()

    try:
        # Inference
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs, 
            max_length     = 2048,
            max_new_tokens = 256,       # 
            # num_beams = 5,              # Use beam search (optional, can help with quality)
            no_repeat_ngram_size = 3,   # Avoid repetitions
            # temperature = 0.7,          # Adjust for diversity in the output
            # early_stopping=True,        # Stop early when the model generates a good output
        )

        # Decode the generated portion
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][input_length:]
        decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        del model
        model = None
        del tokenizer
        tokenizer = None
        del device
        device = None

        return decoded_output
    
    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory error with max_length={256}. Try reducing it.")
        return None
    
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None
    

if __name__ == "__main__":
    decoded_output = inference("The color of the sun is")
    print(f"Generated text: {decoded_output}")
