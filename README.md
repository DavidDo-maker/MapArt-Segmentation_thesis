# MapArt Segmentation

> **A Deep Learning Framework for Map Art Segmentation**
> Bachelor's Thesis Project — International University, VNU-HCM
> **Author:** Đỗ Anh Quân — Student ID: ITCSIU21027

---

## Overview

**MapArt Portrait Extraction** is a deep learning framework for segmenting map-art portraits from complex backgrounds. The project trains and evaluates a **ZoomNeXt**-based segmentation model (with a **ResNet-50** backbone) under several loss-function combinations — Dice, Boundary, BCE, and UAL — and compares their performance on a custom MapArt dataset.

The repository contains everything needed to reproduce the experiments: training scripts, evaluation scripts, configuration files, and three pre-trained model weights corresponding to different loss-function configurations.

## Features

- End-to-end training and evaluation pipeline for map-art segmentation
- **ZoomNeXt** architecture with a **RN50** (ResNet-50) backbone
- Three pre-trained models with different loss function combinations:
  - `Boundary + BCE + UAL`
  - `Dice + BCE + UAL`
  - `Dice + Boundary + BCE + UAL`
- Automatic experiment logging — each run is saved as a separate `exp_*` folder
- Binary-mask outputs that can be compared against ground-truth masks

## Repository Structure

```
MapArt-Segmentation_thesis/
├── configs/
│   └── icod_train.py            # Main training/evaluation config
├── datasets/                    # (downloaded separately — see below)
│   ├── GT/
│   │   ├── Train/               # Ground-truth masks for training
│   │   └── Test/                # Ground-truth masks for testing
│   └── Images/
│       ├── Train/               # Training images
│       └── Test/                # Testing images
├── weights/                     # (downloaded separately — see below)
│   ├── Boundary,BCE,UAL.pth
│   ├── Dice,BCE,UAL.pth
│   └── Dice,Boundary,BCE,UAL.pth
├── outputs/                     # Auto-generated experiment results
├── dataset.yaml                 # (downloaded separately — see below)
├── main_for_image.py            # Main entry point (train / evaluate)
└── requirements.txt
```

## System Requirements

- **Python:** 3.10.2 (recommended)
- **PyTorch:** 2.1.2
- **torchvision:** 0.16.2
- A CUDA-capable GPU is strongly recommended for training (estimated training time: ~6–7 hours).

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/DavidDo-maker/MapArt-Segmentation_thesis.git
cd MapArt-Segmentation_thesis
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install torch==2.1.2
pip install torchvision==0.16.2
```

### 3. Download the dataset and pre-trained weights

The dataset and pre-trained model weights are hosted on Google Drive:

**📁 [Download datasets and weights from Google Drive](https://drive.google.com/drive/folders/196Kx_yLe93HC95nmomP4x83_9seBgALh?usp=drive_link)**

After downloading the `datasets and weight` zip file:

1. Extract the archive.
2. Move the `datasets/` and `weights/` folders into the project root.
3. Move the `dataset.yaml` file into the project root as well.

Your project directory should now match the structure shown above.

## Usage

### Evaluating a Pre-Trained Model

Three pre-trained models are provided, each trained with a different combination of loss functions. Run any of the commands below to evaluate one of them:

**Model 1 — Boundary + BCE + UAL**
```bash
python main_for_image.py --config configs/icod_train.py --model-name RN50_ZoomNeXt --evaluate --load-from weights/Boundary,BCE,UAL.pth --save-results
```

**Model 2 — Dice + BCE + UAL**
```bash
python main_for_image.py --config configs/icod_train.py --model-name RN50_ZoomNeXt --evaluate --load-from weights/Dice,BCE,UAL.pth --save-results
```

**Model 3 — Dice + Boundary + BCE + UAL**
```bash
python main_for_image.py --config configs/icod_train.py --model-name RN50_ZoomNeXt --evaluate --load-from weights/Dice,Boundary,BCE,UAL.pth --save-results
```

### Viewing Results

Each run creates a new experiment folder under `outputs/`, named sequentially (`exp_0`, `exp_1`, `exp_2`, …).

Predicted binary masks are stored in:

```
outputs/exp_<N>/pre/
```

To assess model performance, compare these predicted masks against the ground-truth masks in `datasets/GT/Test/`.

## Training From Scratch (Optional)

If you want to retrain the model:

### Step 1 — Start training

```bash
python main_for_image.py --config configs/icod_train.py --model-name RN50_ZoomNeXt
```

> ⏱️ **Estimated training time:** 6–7 hours on a single GPU.

### Step 2 — Locate the trained weights

After training completes, navigate to the latest experiment folder under `outputs/` (e.g., `outputs/exp_3/`) and open the `pth/` subfolder. Your final trained model is saved as:

```
state_final.pth
```

### Step 3 — Move the trained model into `weights/`

Copy (or move) `state_final.pth` into the `weights/` folder. You may rename it to something more descriptive, e.g. `MyCustomModel.pth`.

### Step 4 — Evaluate your trained model

```bash
python main_for_image.py --config configs/icod_train.py --model-name RN50_ZoomNeXt --evaluate --load-from weights/<your_model_name>.pth --save-results
```

## Pre-Trained Models Summary

| Model File                          | Loss Functions                  | Evaluation Command |
| ----------------------------------- | ------------------------------- | ------------------ |
| `Boundary,BCE,UAL.pth`              | Boundary + BCE + UAL            | Command 1          |
| `Dice,BCE,UAL.pth`                  | Dice + BCE + UAL                | Command 2          |
| `Dice,Boundary,BCE,UAL.pth`         | Dice + Boundary + BCE + UAL     | Command 3          |

These models are included so reviewers can reproduce the thesis results without retraining.

## Author

**Đỗ Anh Quân**
Student ID: **ITCSIU21027**
International University — Vietnam National University, Ho Chi Minh City

## Acknowledgments

This project builds on the **ZoomNeXt** architecture and adapts it for the task of map-art segmentation. Thanks to the thesis advisors and reviewers for their guidance throughout this work.
