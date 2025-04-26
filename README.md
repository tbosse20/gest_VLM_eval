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
1. Run `scripts/video_to_frames.py` to convert videos to frames.
    - Makes sibling folder for frames with "`_frames`".

1. Activate conda env. `conda activate my_env`.

2. Run `scripts/caption_w_models.py` to generate captions across:
    - `--data_folder` manual or found in `config.directories.DATA_FOLDER`. !!!1!!!TODO!!!!!!!
    - Models in the folder `models/` *(note: might crash if not reboot between models. Use `archive`)*.
    - Prompt types in `config/prompts`.

3. Run `scripts/compare_captions.py` to compare with the ground truth.
    - Ground truth captions in `../actedgestures/labels/`.
    - CSV format: `video_name, frame_idx, label`

1. For categories, run `scripts/plot_matrix.py`
    - `--metrics_folder` path to csv results. Manual or found in `config.directories.OUTPUT_FOLDER_PATH`.

4. Run `scripts/plot_metrics.py` to plot and print result to `results/figures`.
    - Use `--prompt_type` or `--gestures` to compare across.

## TODO
- Change metrics validation to boxplot
- Add more sentences and scenarios to metrics validation (impl. multi scenario)
- 