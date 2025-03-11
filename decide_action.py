import llama2
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

    # Generate csv file if not exists
    if not os.path.exists(new_csv_path):
        df = pd.DataFrame(columns=df.columns.tolist() + ["pred_action", "pred_action_id"])
        df = df.drop(columns=["caption"]) #drop caption column
        df.to_csv(new_csv_path, mode="w", index=False, header=True)

    llama2_package = llama2.load_model()

    for idx, row in df.iterrows():
        caption = row["caption"]
        scene_prompt = " ".join([prompts.init_prompt, caption, prompts.task_prompt])
        action = llama2.inference(scene_prompt, llama2_package)
        action = re.sub(r'[\n\r]+', ' ', action).strip()
        action = re.sub(r'\s+', ' ', action).strip()

        # Extract action and action_id
        split_action = action.split(". ")
        row["pred_action"] = split_action[1]
        row["pred_action_id"] = int(split_action[0])

        # Drop caption column
        row = row.drop(labels="caption")
        row.to_frame().T.to_csv(new_csv_path, mode='a', index=False, header=False, encoding="utf-8")

    llama2.unload(llama2_package)

if __name__ == "__main__":
    video_path = "data/sanity/video_0153.mp4"
    csv_path = "data/sanity/caption.csv"
    decide_action_csv(csv_path)
