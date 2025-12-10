# Quaternion Mamba Fusion

Based on Quaternion Mamba (Q-Mamba) for Multi-Modal Image Fusion.

## Introduction

This project implements a multi-modal image fusion system (specifically for IR and RGB image fusion) using Quaternion Mamba blocks. By leveraging the properties of quaternions, the model can effectively handle the inherent correlation between color channels (RGB) and thermal information (IR).

## Features

- **Multi-Modal Fusion**: Fuses single-channel IR images with 3-channel RGB images.
- **Quaternion Representation**: Utilizes Quaternion Neural Networks (QNN) to process 4-channel data (1 IR + 3 RGB) holistically.
- **Mamba Architecture**: Integrates State Space Models (SSM) for efficient long-range dependency modeling.

## Project Structure

- `quaternion_mamba/`: Core package containing models and utilities.
  - `models/`: Model definitions (Q-SSM, Q-Mamba blocks, Fusion Model).
  - `quaternion/`: Quaternion operations and layers.
  - `data/`: Data loading and transforms.
  - `losses/`: Loss functions.
- `scripts/`: Training and testing scripts.
- `configs/`: Configuration files.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

(To be added)
