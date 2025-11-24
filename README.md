# CT-Based Multimodal AI for CSPH Prediction


## 📌 Introduction
**Deep Learning-based Multimodal Framework for Predicting Clinically Significant Portal Hypertension (CSPH) in Cirrhosis.**

This repository contains the official implementation of the paper:  
**"CT-Based Multimodal AI Accurately Predicts Clinically Significant Portal Hypertension in Cirrhosis"** (Submitted to *Hepatology*).

We propose a multimodal AI model that integrates **Abdominal CT images** (visual features) and **Clinical Laboratory Data** (clinical features) to non-invasively predict CSPH (HVPG $\ge$ 10 mmHg).

## 🏗️ Model Architecture

The framework consists of three main components:
1.  **CT Encoder:** A Swin-transformer (e.g., ResNet/EfficientNet) extracting features from liver/spleen segmentation.
2.  **Clinical Encoder:** A Bootstrapping LanguageImage Pre-training for unified vision-language understanding and generation (e.g., Platelet count, Albumin, TB).
3.  **Fusion Module:** A cross-attention mechanism combining visual and clinical embeddings.

## 📂 Directory Structure
```bash
.
├── data/                   # Data preprocessing scripts
│   ├── ct_preprocessing.py
├── models/                 # Model definitions
│   ├── SwinUNETR.py
│   ├── unet3d.py
│   └── fusion_model.py
├── pretrained_weights/
│   ├── supervised_suprem_swinunetr_2100.pth
├── utils/                  # Utility functions (metrics, visualization)
│   ├── datasets.py
│   ├── text_embeddings.py
│   ├── text_utils.py
│   ├── metrics.py
│   └── transforms.py
├── train.py                # Training script
├── inference.py            # Inference/Testing script
├── configs/                # Hyperparameters configurations
│   ├── __init__.py
│   ├── train_config.py
│   └── inference_config.py
└── requirements.txt        # Dependencies
