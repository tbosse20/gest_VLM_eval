# VLM traffic gesture evaluation

## Caption and classification

## Reconstruction
Reconstruction has its own folder.

## Datasets
Link: <LINK>
- Acted Traffic Gestures (ATG)
- Instructive Traffic Gestures In-The-Wild (ITGI)
- Conflicting Authorities and Navigation Gestures (CANG)

## Content
The system generates captions across each of these elements for each video interval:
- Models in `models/`, Qwen, VideoLLaMA2, VideoLLaMA3
- Prompts in `config/prompts.py` "Blank", "Determine", "Body", "Context", "Objective",

## Run - VLM generate captions and compare with ground truth.
1. Set path to data folder `DATA_FOLDER_PATH` in `config.directories`.

1. Run `scripts/video_to_frames.py` to convert videos to frames.
    - Uses `config.directories.VIDEO_FOLDER` or input.
    - Makes sibling folder for frames with "`_frames`".

1. Activate conda env. `conda activate my_env`.

2. Run `scripts/caption_w_models.py` to generate captions across:
    - `--data_folder` manual or found in `config.directories.DATA_FOLDER`. !!!1!!!TODO!!!!!!!
    - Models in the folder `models/` *(note: might crash if not reboot between models. Use `archive`)*.
    - Prompt types in `config/prompts`.

3. Results

    - **Captions Evaluation**
        1. Run `scripts/compare_captions.py` to compare predictions with ground truth.
            - Ground truth captions are located in `../actedgestures/labels/`
            - CSV format: `video_name, frame_idx, label`

        1. Run `scripts/plot_metrics.py` to generate plots and print results to `results/figures/`.
            - Use `--prompt_type` or `--gestures` to customize comparisons.

    - **Category Evaluation**
        1. Run `scripts/plot_matrix.py` for confusion matrix or class-wise accuracy.
            - Use `--metrics_folder` to specify the path to CSV results.
              (This can be manual or found via `config.directories.OUTPUT_FOLDER_PATH`)


## TODO
- Change metrics validation to boxplot
- Add more sentences and scenarios to metrics validation (impl. multi scenario)
- 