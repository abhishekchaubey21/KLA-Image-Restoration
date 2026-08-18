# KLA Hackathon — AI-Based Restoration of Degraded Images

## Overview

This project addresses the problem of restoring degraded semiconductor inspection images using deep learning.

Semiconductor inspection images can suffer from noise, blur, and reduced spatial resolution, making small structures and defects difficult to observe. Our solution uses a supervised deep-learning image-restoration model that learns from paired degraded and high-quality images.

The system takes a degraded/low-resolution image as input and produces a cleaner, sharper and higher-resolution restored image.

### Pipeline

Degraded / Noisy Image
        ↓
Deep Learning Restoration Model
        ↓
Denoising + 2× Restoration / Super-Resolution
        ↓
Restored Image
        ↓
PSNR / SSIM / LPIPS Evaluation


## 1. Dataset

The training dataset contains paired degraded images and corresponding high-quality ground-truth images.

### Dataset components

- `NoisyLR` — degraded / low-resolution input images
- `GT 0` — ground-truth images
- `GT 1` — additional ground-truth images

The dataset contains:

- **3200 matched image pairs**
- **2880 training samples**
- **320 validation samples**

The degraded input images are stored as `.npy` arrays.

The model operates on grayscale images (`in_nc=1`).

### Train / Validation split

The dataset loader automatically creates a held-out validation set so that model performance can be measured on images that were not used for parameter updates.


## 2. Model

We use a compact RRDB-based image-restoration network.

The `small` configuration was selected because the model needs to train on a laptop GPU with 6 GB VRAM.

### Model configuration

- Input channels: `1`
- Output channels: `1`
- Scale factor: `2`
- Model size: `small`
- Parameters: approximately **2.55 million**

The model learns a mapping:

`Degraded Image → Restored High-Resolution Image`

The network is trained end-to-end using the paired degraded and ground-truth images.


## 3. Training

Training was performed using:

- NVIDIA RTX 3050 Laptop GPU
- 6 GB VRAM
- Mixed precision (BF16)
- AdamW optimizer
- Gradient clipping
- Cosine learning-rate scheduling
- EMA (Exponential Moving Average) model weights

### Initial training

The first training stage used the complete training pipeline and established the baseline model.

The best model from this stage was then used as the starting point for fine-tuning.

### Fine-tuning

The final fine-tuning experiment used:

- Learning rate: `2e-5`
- Batch size: `6`
- SSIM loss weight: `0.4`
- LPIPS loss weight: `0.08`
- Extra synthetic noise augmentation: disabled
- Model: `small`
- Input channels: `1`
- Scale: `2`

Instead of training from random initialization, the previously trained best model was loaded and refined using the lower learning rate.

The final checkpoint is:

`checkpoints/best_model.pt`


## 4. Loss Functions

The training objective combines multiple losses so that the restored image is accurate both at the pixel level and in terms of image structure/perceptual quality.

### Pixel Loss

The pixel-level component encourages the predicted image to remain close to the ground-truth image.

### SSIM Loss

SSIM focuses on structural similarity between the restored and ground-truth images.

This is particularly useful for image restoration because preserving edges, patterns and local structures is important.

### LPIPS Loss

LPIPS is a perceptual similarity metric used as an additional training objective.

It encourages the restored image to have perceptually meaningful features similar to the ground-truth image.

The final fine-tuning configuration used:

- Pixel loss weight: `1.0`
- SSIM weight: `0.4`
- LPIPS weight: `0.08`


## 5. Evaluation Metrics

Three main metrics are used to evaluate restoration quality.

### PSNR

Peak Signal-to-Noise Ratio measures pixel-level reconstruction quality.

Higher PSNR is better.

### SSIM

Structural Similarity Index measures how similar the structure of the restored image is to the ground truth.

Higher SSIM is better.

### LPIPS

LPIPS measures perceptual similarity between images.

Lower LPIPS is better.


## 6. Final Validation Results

The best validation checkpoint obtained during fine-tuning achieved:

| Metric | Best Validation Result |
|---|---:|
| PSNR | **≥ 28.67 dB** |
| SSIM | **≥ 0.8041** |
| LPIPS | **≤ 0.16** |

The best checkpoint was selected based on validation SSIM.

The final best model was saved as:

`checkpoints/best_model.pt`


## 7. EMA

An Exponential Moving Average (EMA) of model weights was maintained during training.

At every validation stage, both:

- the current/raw model
- the EMA model

were evaluated.

The version with the higher validation SSIM was selected as the best checkpoint.

This helps reduce instability caused by small epoch-to-epoch changes in validation performance.


## 8. Standalone Inference

`evaluate_submission.py` is the standalone inference script.

It:

1. Loads the trained checkpoint.
2. Finds all supported input files.
3. Loads the degraded images.
4. Groups images by spatial dimensions.
5. Runs batched inference.
6. Restores the images using the trained model.
7. Converts the outputs to PNG.
8. Saves the restored images to the specified output directory.
9. Reports end-to-end inference time.

The script is designed to run without manual modification after the input and output paths are supplied.


## 9. Inference Command

The inference script can be executed using:

```bash
python evaluate_submission.py \
    --input_dir path/to/test_images \
    --output_dir path/to/restored_outputs \
    --checkpoint checkpoints/best_model.pt
