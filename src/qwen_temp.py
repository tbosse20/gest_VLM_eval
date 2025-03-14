from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import pandas as pd
import os
from tqdm import tqdm

def from_end_frame(video_folder, start_frame, interval, end_frame):
    return [
        f"{video_folder}/frame_{frame_count:04d}.png"
        for frame_count in range(start_frame, end_frame, interval)
    ]

def from_n_frame(video_folder, start_frame, interval, n_frames):
    return [
        f"{video_folder}/frame_{start_frame + frame_count:04d}.png"
        for frame_count in range(0, n_frames, interval)
        if os.path.exists(f"{video_folder}/frame_{start_frame + frame_count:04d}.png")
    ]

def generate_frame_list(video_folder, start_frame, interval, end_frame, n_frames):
    if end_frame is not None: 
        return from_end_frame(video_folder, start_frame, interval, end_frame)
    if n_frames is not None: 
        return from_n_frame(video_folder, start_frame, interval, n_frames)

def inference(
        video_folder,
        start_frame = 0,
        interval = 1,
        end_frame = None,
        n_frames = None,
    ):

    frame_list = generate_frame_list(video_folder, start_frame, interval, end_frame, n_frames)
    if len(frame_list) == 0:
        return 'empty'
    
    # Messages containing a images list as a video and a text query
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_list,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Analyze the provided image and describe the pedestrian in detail, including their appearance, clothing, posture, and any significant features. Additionally, interpret their gesture and body language—are they signaling, pointing, waving, or displaying any other meaningful action? Based on their gesture, infer their possible intent or communication (e.g., are they trying to cross the street, signal a vehicle, or interact with someone?). Provide a clear and concise description."},
            ],
        }
    ]
    # # Messages containing a video and a text query
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "file:///path/to/video1.mp4",
    #                 "max_pixels": 360 * 420,
    #                 "fps": 1.0,
    #             },
    #             {"type": "text", "text": "Describe this video."},
    #         ],
    #     }
    # ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # print("Start_frame:", start_frame)
    # print(output_text)
    # print()

    return output_text[0]

# inference(
#     video_folder="data/sanity/input/video_0153",
#     start_frame=50,
#     n_frames=8,
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype=torch.bfloat32,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

csv_path = "data/sanity/output/caption_man_window=8_explain.csv"
columns = ["video_name", "frame_idx", "caption"]
# Generate csv file if not exists
if not os.path.exists(csv_path):
    df = pd.DataFrame(columns=columns)
    df.to_csv(csv_path, mode="w", index=False, header=True)

window = 8 # Batch size (<16)
for i in tqdm(range(0, 160 - window, window), desc="Processing"):
    respond = inference(
        video_folder="data/sanity/input/video_0153/pedestrian_man",
        start_frame=i,
        n_frames=window,
    )

    df = pd.DataFrame({"video_name": ['video_0153_man'], "frame_idx": [i], "caption": [respond]})
    df.to_csv(csv_path, mode="a", index=False, header=False)
