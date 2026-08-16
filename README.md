# KLA Hackathon — AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview

This project implements an AI-based image restoration pipeline for degraded semiconductor inspection images.

The model takes a degraded/noisy low-resolution image as input and generates a cleaner, sharper and higher-resolution restored image. The restoration is learned using paired degraded images and corresponding high-quality ground-truth images.

The overall pipeline is:

```text
Degraded / Noisy Low-Resolution Image
                ↓
       AI Restoration Model
                ↓
     Restored High-Resolution Image
                ↓
       Comparison with Ground Truth
```

The goal is to recover useful spatial details while reducing degradation in semiconductor inspection images.

---

## Dataset

The dataset consists of paired degraded and high-quality images.

- `NoisyLR` — degraded/low-resolution input images
- `GT 0` and `GT 1` — corresponding high-quality ground-truth images

A total of **3200 matched image pairs** were used:

- **2880 pairs** for training
- **320 pairs** for validation

The input images are grayscale `128×128` images, while the corresponding ground-truth images provide the higher-resolution target used for supervised training.

During training, the model learns to transform the degraded input into an output that is as close as possible to its corresponding ground-truth image.

---

## Model

We use a compact deep-learning image restoration network designed to run efficiently on limited GPU memory.

The final model configuration is:

- **Input channels:** 1 (grayscale)
- **Output channels:** 1
- **Scale factor:** 2×
- **Model size:** Small
- **Parameters:** approximately 2.55 million
- **Input size:** 128×128
- **Output size:** 256×256

The model learns the following mapping:

```text
Degraded 128×128 Image
          ↓
   Restoration Network
          ↓
Restored 256×256 Image
```

The network performs image restoration, denoising and spatial upscaling in a single pipeline.

---

## Training

The model is trained using supervised learning on the paired degraded and ground-truth images.

For each training sample:

1. The degraded image is given to the model.
2. The model generates a restored image.
3. The restored image is compared with the ground-truth image.
4. The losses are calculated.
5. Backpropagation updates the model weights.
6. The model is evaluated on the validation set.
7. The best-performing checkpoint based on validation SSIM is saved.

### Loss Functions

The training objective combines multiple losses:

- **Pixel reconstruction loss** — encourages the generated pixels to remain close to the ground truth.
- **SSIM loss** — helps preserve structural information, edges and image details.
- **LPIPS loss** — encourages perceptually meaningful image reconstruction.

For the final fine-tuning run, we used:

- **Learning rate:** `2e-5`
- **SSIM weight:** `0.4`
- **LPIPS weight:** `0.08`
- **Additional synthetic noise augmentation:** disabled
- **Mixed precision:** BF16
- **Gradient clipping:** `1.0`

---

## Validation Results

The model was evaluated on the held-out validation set using three image-quality metrics:

- **PSNR** — measures pixel-level reconstruction quality. Higher is better.
- **SSIM** — measures structural similarity between the restored image and ground truth. Higher is better.
- **LPIPS** — measures perceptual difference between the images. Lower is better.

The best validation result obtained was:

| Metric | Best Result |
|---|---:|
| SSIM | **0.7741** |
| PSNR | **28.17 dB** |
| LPIPS | **~0.19** |

The trained model weights are provided as:

```text
checkpoints/best_model.pt
```

---

## Repository Structure

```text
KLA-Image-Restoration/
│
├── checkpoints/
│   └── best_model.pt
│
├── dataset.py
├── evaluate_submission.py
├── losses.py
├── metrics.py
├── model.py
├── requirements.txt
└── train.py
```

### File Description

- `model.py` — defines the image restoration neural network
- `dataset.py` — loads and prepares the degraded and ground-truth image pairs
- `losses.py` — implements the pixel, SSIM and LPIPS loss functions
- `metrics.py` — calculates PSNR, SSIM and LPIPS evaluation metrics
- `train.py` — contains the complete training and validation pipeline
- `evaluate_submission.py` — standalone inference script for generating restored images
- `checkpoints/best_model.pt` — contains the trained model weights
- `requirements.txt` — contains the Python dependencies required to run the project

---

## Environment Setup

Create a Python virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Verify that PyTorch can detect the GPU:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Running Inference

The standalone evaluation script accepts:

- Input image directory
- Output directory
- Trained model checkpoint

Run:

```bash
python evaluate_submission.py --input_dir PATH_TO_TEST_IMAGES --output_dir restored_test_outputs --checkpoint checkpoints/best_model.pt
```

For example:

```bash
python evaluate_submission.py --input_dir test_images --output_dir restored_test_outputs --checkpoint checkpoints/best_model.pt
```

The evaluation script:

1. Loads the trained model.
2. Reads all supported input images/arrays.
3. Groups inputs by spatial dimensions.
4. Runs batched GPU inference.
5. Generates restored images.
6. Saves the restored images as PNG files.

The evaluation script is designed to run without manual modification.

---

## Inference Performance

The final evaluation pipeline was tested locally on an NVIDIA RTX 3050 Laptop GPU using 3200 input images.

The test produced:

```text
Processed images:       3200
Model load time:        0.228 s
Inference + I/O time:   64.081 s
Total end-to-end time:  64.309 s
```

The evaluation pipeline is optimized for GPU inference and is designed to run on the KLA H100 benchmarking environment.

---

## Reproducing Training

Training can be reproduced using:

```bash
python train.py --gt_dirs data/GT_0 data/GT_1 --lr_dir data/NoisyLR --epochs 60 --batch_size 8 --model_size small --in_nc 1 --use_amp
```

The training pipeline performs:

- Paired dataset loading
- Data augmentation
- Model training
- Combined loss optimization
- Validation
- PSNR/SSIM/LPIPS evaluation
- Best-model checkpointing

---

## Output

For each degraded input image, the inference script produces a corresponding restored PNG image.

The restoration pipeline can be represented as:

```text
Noisy / Low-Resolution Image
              ↓
       AI Restoration Model
              ↓
Restored / Higher-Resolution Image
```

The restored images can be evaluated against the available ground-truth images using PSNR, SSIM and LPIPS.

---

## Team Objective

The objective of this implementation is to improve the quality and usability of degraded semiconductor inspection images by reducing degradation, recovering spatial details and producing higher-resolution images suitable for downstream inspection and analysis.
