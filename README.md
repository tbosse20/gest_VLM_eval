# RMPS

# Caption

1. Convert each video to a folder contaning all frames
2. Run `scripts/caption_w_model` with fixed variables:
- `video_frames` - Path to frames folders
- `window` - Batch of frames
- `models_folder` - Path to model modules
    Captions all videos' frames across the variables:
    - Differnt prompt types in `config.prompts`
    - Fixed `window`  
3. Run `results