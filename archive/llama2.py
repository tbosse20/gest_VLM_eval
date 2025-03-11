import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

def load_model():

    gc.collect()
    torch.cuda.empty_cache()

    print('Loading model: Llama-3-8B')

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
    model.eval()

    # If you are not fine-tuning, disable gradient computation for inference
    for param in model.parameters():
        param.requires_grad = False

    return model, tokenizer, device

def unload(llama_package):
    model, tokenizer, device = llama_package

    del model
    del tokenizer
    del device

    gc.collect()
    torch.cuda.empty_cache()

def inference(prompt: str, llama_package=None):

    model, tokenizer, device = llama_package or load_model()

    try:
        # Inference
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                # max_length     = 2048,
                max_new_tokens = 8,       # 
                # num_beams = 5,              # Use beam search (optional, can help with quality)
                no_repeat_ngram_size = 3,   # Avoid repetitions
                temperature = 0.7,          # Adjust for diversity in the output
                early_stopping=True,        # Stop early when the model generates a good output
            )

            # Decode the generated portion
            input_length = inputs["input_ids"].shape[-1]
            generated_tokens = outputs[0][input_length:]
            decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

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
