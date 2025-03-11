import llama_instruct
import prompts
import gc
import os
import pandas as pd
import prompts
import re
import torch

def decide_action_csv(csv_path):
    torch.cuda.memory_summary(device=None, abbreviated=False)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)

    # Load df
    df = pd.read_csv(csv_path)

    # Make new csv file name
    new_csv_path = "pred_actions.csv"
    directory = os.path.dirname(csv_path)
    new_csv_path = os.path.join(directory, new_csv_path)

    # Generate csv file if not exists
    if not os.path.exists(new_csv_path):
        df = pd.DataFrame(columns=df.columns.tolist() + ["pred_action", "pred_action_id"])
        df = df.drop(columns=["caption"]) #drop caption column
        df.to_csv(new_csv_path, mode="w", index=False, header=True)

    llama2_package = llama_instruct.load_model()

    for idx, row in df.iterrows():
        caption = row["caption"]
        
        # Construct message/prompt
        messages = [
            {"role": "system", "content": prompts.setting_prompt},
            {"role": "user", "content": prompts.task_prompt + caption + prompts.output_prompt},
        ]
        json_data = llama_instruct.inference(messages, llama2_package)

        # Extract action and action_id
        action = json_data['action']
        row["pred_action"] = action

        # Drop caption column
        row = row.drop(labels="caption")
        row.to_frame().T.to_csv(new_csv_path, mode='a', index=False, header=False, encoding="utf-8")

    llama_instruct.unload(llama2_package)

if __name__ == "__main__":
    video_path = "data/sanity/video_0153.mp4"
    csv_path = "data/sanity/caption.csv"
    decide_action_csv(csv_path)
