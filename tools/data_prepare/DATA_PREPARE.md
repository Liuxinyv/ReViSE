# Data Preparation

This document describes the data preparation pipeline for training ReViSE models. The process involves extracting latent features from videos and text prompts to create efficient training datasets.

## Overview

To finetune models, input videos and their corresponding prompts need to be preprocessed into latent features and multimodal language model (MLM) features. This offline feature extraction approach significantly reduces GPU memory requirements during training and improves overall training efficiency.

The data preparation consists of two main steps:

## Step 1: VAE and T5 Feature Extraction

### Overview
Extract video latent features using a VAE (Variational Autoencoder) and text embeddings using T5 encoder.

### Usage
```bash
bash tools/data_prepare/run_vae_feature_edit.sh
```

### Input Format
The input should be a JSON file containing video paths and corresponding prompts. Each entry should follow this structure:

```json
[
    {
    "source_video_path": "094db9758726f58e839ac7d2a21b93a9_3.mp4",
    "target_video_path": "094db9758726f58e839ac7d2a21b93a9_0.mp4",
    "instruction": "What if the cold air meets the wet ground and causes moisture to condense into a low mist?",
    "type": "Existential Reasoning Edits"
  },
  ...
]
```

### Important Notes
- Only the `video` field and the `value` field from the `gpt` conversation are required
- Other fields in the JSON structure are ignored during processing
- Each video-prompt pair will be saved as a separate pickle file containing the extracted features

### Output
The script generates pickle files containing:
- VAE latent features for video frames
- T5 text embeddings for prompts
- Metadata including video path and frame information

## Step 2: AR Model Feature Extraction

### Overview
Extract autoregressive (AR) model features from the VAE features generated in Step 1.

### Usage
```bash
bash tools/data_prepare/run_ar_feature_edit.sh
```

### Input
The script uses the pickle file list generated from Step 1 as input. Set the `$DATA_FILE` variable in `run_ar_feature_edit.sh` to point to your VAE feature file list.

### Output
Final pickle files containing all features required for training:
- VAE latent features
- T5 text embeddings  
- AR model features
- Complete metadata

## Configuration

### Key Parameters
- **Frame Count**: Number of frames to extract (default: 81, must be 4n+1)
- **Sampling Rate**: Frame sampling interval (default: 3)
- **Target Size**: Output resolution (default: 480x832)
- **Skip Frames**: Number of initial frames to skip (default: 0)

## Integration with Training

The generated pickle files can be directly used with the training pipeline. The feature extraction process ensures optimal memory usage and training efficiency.