# PneumoniadetectioninViT
Implementation of Vision Transformer architecture for medical image classification, trained on the Chest X-Ray Pneumonia dataset with TensorFlow/Keras.

---
# Overview
This project implements a **Vision Transformer (ViT)** model for automated detection of **pneumonia from chest X-ray images** using deep learning.  
The model is trained on the popular Kaggle dataset **Chest X-Ray Images (Pneumonia)** and performs binary classification:
- NORMAL
- PNEUMONIA

The project demonstrates how transformer-based architectures can be applied to medical image classification tasks.

---
# Model Architecture
The model is based on the **Vision Transformer (ViT)** concept:

### Pipline:
1. Input image resizing
2. Patch extraction
3. Linear patch embedding
4. Positional encoding
5. Multi-head self-attention blocks
6. Transformer encoder layers
7. MLP classification head
8. Softmax output layer

### Core Components:
- Patch Embedding
- Multi-Head Attention
- Transformer Encoder
- MLP Head

---
## Dataset

**Source:** Kaggle  
**Dataset Name:** Chest X-Ray Images (Pneumonia)  
**Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---
## Technologies Used

- Python 3
- TensorFlow / Keras
- Vision Transformer (ViT)
- NumPy
- Matplotlib
- Kaggle API
- Google Colab

---

