# VLM traffic gesture evaluation

## Datasets
- ATG
- ITGI

## Content
- Models in `models/`, Qwen, VideoLLaMA2, VideoLLaMA3
- Prompts in `config/prompts.py` "Blank", "Determine", "Body", "Context", "Objective",

## Run - VLM generate captions and compare with ground truth.
1. Run `dev/split_video.py` to convert videos to frames.
2. Run `scripts/caption_w_models.py` to generate captions across:
    - Models in the folder `models/`.
    - Prompt types in `config/prompts`.
3. Run `results/scripts/compare_captions.py` to compare with the ground truth.
    - Annotate the ground truth captions in `data/labels/`.
        - Format: `video_name, frame_idx, label`
4. Run `results/scripts/plot_metrics.py` to boxplot and print result.
    - Use `--prompt_type` or `--gestures` to compare across.