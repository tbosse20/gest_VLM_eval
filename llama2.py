import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

torch.cuda.empty_cache()

model_name = "meta-llama/CodeLlama-7b-Instruct-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
else:
    device = torch.device("cpu")
    
def inference(prompt: str):

    # Inference
    inputs = tokenizer(prompt, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = inputs.to(device)

    outputs = model.generate(
        **inputs, 
        # max_length = 100,           # Define the maximum output length
        num_beams = 5,              # Use beam search (optional, can help with quality)
        no_repeat_ngram_size = 2,   # Avoid repetitions
        temperature = 0.7,          # Adjust for diversity in the output
        early_stopping=True,        # Stop early when the model generates a good output
    )

    # Decode the generated portion
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print(f"Generated text: {decoded_output}")

    return decoded_output

if __name__ == "__main__":
    inference("The color of the sun is")
