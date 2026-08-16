# KLA Hackathon 2026 — AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview

This project implements a compact AI-based image restoration pipeline for degraded semiconductor inspection images, developed for **SEMICON India Hackathon 2026 — KLA Problem Statement 1**.

The model takes a degraded, noisy and low-resolution grayscale inspection image and produces a cleaner, sharper and higher-resolution restoration. The network jointly performs **degradation removal and 2× spatial upscaling** rather than using separate sequential models.

The overall pipeline is:

```text
Degraded / Noisy Low-Resolution Image
                ↓
       Compact RRDB Restoration Network
                ↓
       2× Super-Resolution Output
                ↓
       Restored High-Resolution Image
                ↓
      PSNR / SSIM / LPIPS Evaluation
```

The objective is to recover useful spatial details while suppressing degradation, while keeping the model compact enough for practical inference.

The approach is designed around the three degradation requirements of the KLA problem:

* Speckle noise
* Gaussian noise
* Spatial-resolution reduction / 2× super-resolution

The model is also designed with inference efficiency and generalization in mind, since KLA evaluates both restoration quality and inference performance.

---

## Problem Context

Semiconductor inspection relies on high-quality microscopic images to identify small structures and defects. Noise and loss of spatial resolution can obscure fine details and reduce the reliability of downstream inspection.

The KLA problem provides paired degraded and ground-truth images and requires a model that learns to reconstruct the high-resolution target from the degraded input.

A key requirement is that the restoration model should handle multiple degradation types **simultaneously**, rather than assuming that only one degradation is present.

---

## Dataset

The development dataset consists of paired degraded and high-quality grayscale images.

* `NoisyLR` — degraded / low-resolution input images
* `GT_0` and `GT_1` — corresponding ground-truth images

A total of **3200 matched samples** were used:

* **2880 samples** for training
* **320 samples** for validation

The primary development configuration uses:

```text
Input:        128 × 128 grayscale
Target:       256 × 256 grayscale
Channels:     1
Scale factor: 2×
```

The model learns the mapping:

```text
NoisyLR 128×128
      ↓
Restoration Network
      ↓
Restored 256×256
      ↓
Ground Truth 256×256
```

The official KLA test set is designed to include both in-distribution and out-of-distribution samples. The official test set was not used for the validation metrics reported below.

---

## Model Architecture

The proposed model is a **compact Residual-in-Residual Dense Block (RRDB) CNN**, inspired by the RRDB architecture used in ESRGAN/Real-ESRGAN and adapted for the computational constraints of this task.

### Configuration

| Parameter         | Configuration |
| ----------------- | ------------- |
| Input channels    | 1             |
| Output channels   | 1             |
| Input resolution  | 128×128       |
| Output resolution | 256×256       |
| Scale factor      | 2×            |
| RRDB blocks       | 6             |
| Feature channels  | 48            |
| Growth channels   | 24            |
| Parameters        | ~2.55M        |

The architecture uses:

* RRDB residual-dense blocks
* Bicubic global residual skip
* PixelShuffle ×2 upsampling
* ICNR initialization
* No final sigmoid activation
* Output clamping during evaluation

### Residual Reconstruction

Instead of learning the complete high-resolution image from scratch, the network uses a bicubic-upsampled image as a global residual baseline:

```text
Input
  │
  ├──────────────→ Bicubic ×2 ───────────────┐
  │                                          │
  ↓                                          ↓
RRDB Restoration Body → PixelShuffle → Learned Residual
                                             │
                                             ↓
                              Bicubic Base + Residual
                                             │
                                             ↓
                                    Restored Output
```

This allows the network to focus primarily on learning the missing correction and fine details.

The implementation also uses ICNR initialization for the pre-PixelShuffle convolution to reduce checkerboard artifacts during sub-pixel upsampling.

---

## Why This Architecture

The architecture was selected primarily because it provides a strong quality-to-complexity trade-off.

### Compact model

The network contains approximately **2.55 million parameters**, allowing training on a laptop RTX 3050 with 6 GB VRAM.

### Joint restoration

Denoising and super-resolution are handled in the same network rather than using a sequential denoiser followed by a super-resolution model. This reduces the possibility of error propagation between separate restoration stages.

### Residual learning

The bicubic global skip provides a strong low-frequency reconstruction while the network focuses on recovering the residual high-frequency information.

### Efficient upsampling

PixelShuffle provides learned 2× upsampling while keeping most feature extraction at the lower spatial resolution.

---

## Training Strategy

The model is trained using supervised learning on paired degraded and ground-truth images.

For each training sample:

1. Load the degraded input.
2. Generate the restored high-resolution image.
3. Compare the output against the corresponding ground truth.
4. Calculate the combined restoration loss.
5. Backpropagate the loss.
6. Update the model parameters.
7. Evaluate the model on the validation set.
8. Save the best-performing checkpoint according to validation performance.

### Degradation-Aware Training

The training pipeline supports synthetic degradation augmentation to improve robustness to different image degradation conditions, including Gaussian and speckle noise.

The final fine-tuning stage used the paired KLA data directly with additional synthetic noise augmentation disabled.

This separates the degradation-robustness training stage from the final paired-data fine-tuning stage.

---

## Loss Functions

The restoration objective combines pixel-level and perceptual/structural objectives.

### Charbonnier / Robust Reconstruction Loss

A robust pixel reconstruction objective is used to reduce sensitivity to outliers while encouraging accurate image reconstruction.

### SSIM Loss

SSIM-based optimization encourages preservation of:

* Structural information
* Edges
* Local contrast
* Fine image structures

### LPIPS Loss

LPIPS provides a perceptual similarity objective that complements pixel-level reconstruction metrics.

The final fine-tuning configuration used:

```text
Learning rate:          2e-5
SSIM weight:            0.4
LPIPS weight:           0.08
Mixed precision:        BF16
Gradient clipping:     1.0
Synthetic noise during
final fine-tuning:      Disabled
```

---

## Validation Results

The model was evaluated on the held-out validation set using the same three image-quality metrics highlighted for the KLA restoration task.

* **PSNR** — pixel-level reconstruction quality; higher is better.
* **SSIM** — structural similarity; higher is better.
* **LPIPS** — perceptual distance; lower is better.

### Best Validation Result

| Metric    |       Result |
| --------- | -----------: |
| **PSNR**  | **28.67 dB** |
| **SSIM**  |   **0.8041** |
| **LPIPS** |    **~0.16** |

These are the results reported for the submitted solution.

> **Checkpoint consistency:** `checkpoints/best_model.pt` should be the exact checkpoint corresponding to the metrics above.

The official KLA test set is separate from this validation set and includes out-of-distribution samples. Therefore, no OOD test score is claimed here unless it has been independently measured on the official test data.

---

## Visual Restoration

The restoration process is:

```text
Degraded Input
      ↓
Noise / Resolution Degradation
      ↓
Compact RRDB Restoration Network
      ↓
Restored Output
      ↓
Comparison with Ground Truth
```

The model is intended to remove degradation while preserving fine spatial structures rather than simply smoothing the image.

---

## Inference

The repository contains a standalone evaluation script:

```text
evaluate_submission.py
```

The script accepts:

* Input image directory
* Output directory
* Model checkpoint

### Run inference

```bash
python evaluate_submission.py \
    --input_dir PATH_TO_TEST_IMAGES \
    --output_dir restored_test_outputs \
    --checkpoint checkpoints/best_model.pt
```

Example:

```bash
python evaluate_submission.py \
    --input_dir test_images \
    --output_dir restored_test_outputs \
    --checkpoint checkpoints/best_model.pt
```

The script:

1. Loads the trained checkpoint.
2. Automatically detects the available device.
3. Reads supported image files and NumPy arrays.
4. Groups inputs by spatial dimensions.
5. Performs batched inference.
6. Generates restored images.
7. Saves the outputs as PNG files.

The script is designed to operate as a standalone inference pipeline without requiring manual modification.

The default inference configuration includes:

```text
Batch size:       64
I/O workers:      8
PNG compression:  1
```

---

## Inference Performance

The pipeline was benchmarked locally on an:

```text
GPU: NVIDIA RTX 3050 Laptop GPU
VRAM: 6 GB
Images: 3200
```

Measured end-to-end performance:

```text
Processed images:       3200
Model load time:        0.228 s
Inference + I/O time:   64.081 s
Total end-to-end time:  64.309 s
```

This corresponds to approximately:

```text
50 images/second
```

for the measured end-to-end local run.

The reported time includes inference and I/O and should not be interpreted as pure neural-network compute time.

The evaluation script is designed for the KLA benchmarking environment, where the submission will be evaluated on an H100 GPU.

---

## Reproducing Training

Training can be reproduced using the provided `train.py` script.

Example:

```bash
python train.py \
    --gt_dirs data/GT_0 data/GT_1 \
    --lr_dir data/NoisyLR \
    --epochs 60 \
    --batch_size 8 \
    --model_size small \
    --in_nc 1 \
    --use_amp
```

The training pipeline includes:

* Paired dataset loading
* Data augmentation
* Restoration model construction
* Combined loss optimization
* Mixed-precision training
* Gradient clipping
* Validation
* PSNR / SSIM / LPIPS evaluation
* Best-model checkpointing

---

## Repository Structure

```text
KLA-Image-Restoration/
│
├── checkpoints/
│   ├── best_model.pt
│   └── .gitkeep
│
├── restored_test_outputs/
│   └── [restored test images]
│
├── dataset.py
├── evaluate_submission.py
├── losses.py
├── metrics.py
├── model.py
├── requirements.txt
├── train.py
└── README.md
```

### File Description

| File                        | Purpose                                                  |
| --------------------------- | -------------------------------------------------------- |
| `model.py`                  | Defines the compact RRDB restoration network             |
| `dataset.py`                | Loads and prepares degraded and ground-truth image pairs |
| `losses.py`                 | Implements the reconstruction, SSIM and LPIPS objectives |
| `metrics.py`                | Calculates PSNR, SSIM and LPIPS                          |
| `train.py`                  | Training and validation pipeline                         |
| `evaluate_submission.py`    | Standalone inference/evaluation script                   |
| `checkpoints/best_model.pt` | Final submitted model checkpoint                         |
| `requirements.txt`          | Pinned Python dependencies                               |
| `restored_test_outputs/`    | Restored outputs generated by the submitted model        |

---

## Environment Setup

Create a Python virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify the PyTorch installation:

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The pinned environment currently includes:

```text
numpy==1.26.4
Pillow==11.3.0
torch==2.11.0
torchvision==0.26.0
lpips==0.1.4
```

---

## Submission Artifacts

For the final hackathon submission, the public repository contains the implementation required to reproduce and evaluate the model:

* `README.md`
* Standalone `evaluate_submission.py`
* Training script `train.py`
* Trained model weights
* `requirements.txt`
* Restored test outputs

The most important file for evaluation is:

```text
evaluate_submission.py
```

It is intentionally provided as a standalone Python script so that the evaluation environment can invoke the model without modifying the source code.

Before final submission, the checkpoint, evaluation script and restored-output folder should all correspond to the same submitted model version.

---

## Technical Design Summary

The main design choices are:

```text
Compact RRDB
      +
Bicubic Global Residual Skip
      +
PixelShuffle ×2
      +
ICNR Initialization
      +
Robust Reconstruction Loss
      +
SSIM Loss
      +
LPIPS Loss
      +
Degradation-Aware Training
      +
Batched GPU Inference
```

The resulting model provides a compact restoration pipeline with approximately **2.55M parameters**, while jointly addressing image degradation and 2× spatial-resolution recovery.

---

## Generalization

The official KLA evaluation includes both in-distribution and out-of-distribution images.

The development pipeline therefore avoids treating the validation set as the official test set. The model is trained with degradation-aware strategies intended to improve robustness to variations in noise and image appearance.

No numerical OOD performance is claimed in this README because the official KLA test set is separate from the development validation set.

---

## Limitations

The reported metrics are validation results and should not be interpreted as official KLA test-set scores.

The current local inference benchmark combines neural-network inference and file I/O:

```text
Inference + I/O = 64.081 s for 3200 images
```

Therefore, the reported 64.309 s end-to-end time is not a pure model-compute benchmark.

Final performance on the official KLA test set, particularly on out-of-distribution samples, will depend on the unseen test distribution and the official H100 evaluation environment.

---

## Research & Technical References

The implementation is based on established image-restoration and image-quality techniques, including:

* Wang et al. — **ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks** (2018) — RRDB architecture
* Lim et al. — **Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)** (2017) — residual/global skip design
* Aitken et al. — **Checkerboard artifact free sub-pixel convolution** (2017) — ICNR initialization
* Wang et al. — **Image Quality Assessment: From Error Visibility to Structural Similarity** (2004) — SSIM
* Zhang et al. — **The Unreasonable Effectiveness of Deep Features as a Perceptual Metric** (2018) — LPIPS
* Charbonnier et al. — robust Charbonnier/pseudo-Huber loss formulation

---

## Demo

A working demonstration video of the training/inference workflow is provided with the hackathon submission.

---

## Team

**Team DrishtiSilicon**

Problem Statement:

**AI-Based Restoration of Degraded Images for Semiconductor Inspection — KLA PS01**

Repository:

**KLA-Image-Restoration**
