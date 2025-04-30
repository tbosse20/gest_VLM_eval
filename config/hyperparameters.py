# Define generation hyperparameters
generation_args = {
    "temperature":           0.2,  # Controls randomness (lower = more deterministic)
    "top_k":                  50,  # Limits token selection to top 50 choices
    "top_p":                0.95,  # Nucleus sampling threshold
    "max_new_tokens":        512,  # Max number of tokens in response
    "repetition_penalty":    1.2,  # Penalizes repetition
    "no_repeat_ngram_size":    2,  # Prevents repeating n-grams (3-grams)
    "length_penalty":        1.0,  # Adjusts output length preference
}

interval = 1  # Interval between frames for captioning  (default=1)
window = 0    # Frame window size for captioning        (default=0, full video)