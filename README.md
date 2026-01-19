# Synthetic Leprosy Image Generation via Transfer Learning from Large Wound Datasets

## Abstract
Dermatological research on rare and neglected diseases such as leprosy is limited by the scarcity of high-quality, annotated clinical images. This project investigates whether generative models trained on large chronic wound datasets and fine-tuned on limited leprosy data can produce realistic synthetic leprosy images to improve data availability for research and downstream machine-learning tasks.

Publicly available wound-image datasets were first preprocessed using a DeepLabV3-ResNet50 segmentation model to isolate lesion regions and standardize image structure. These segmented images were used to train a diffusion-based generative model to learn general skin-lesion texture, color, and morphological patterns. The model was subsequently fine-tuned on a substantially smaller leprosy dataset to transfer disease-specific visual characteristics, including lesion boundaries and pigmentation.

Generated images were evaluated using qualitative visual inspection and distribution-based similarity metrics. Results indicate that fine-tuning on limited leprosy data enables the model to generate diverse and realistic synthetic images that capture key morphological features of leprosy lesions. This work demonstrates the potential of diffusion-based generative models as a data-augmentation strategy for mitigating dataset scarcity in dermatological machine learning.

## Research Question
Can generative models pretrained on large wound-image datasets and fine-tuned on limited leprosy data generate realistic synthetic leprosy images suitable for data augmentation and downstream model training?

## Methods Overview
- Wound and lesion segmentation 
- ROI extraction and image standardization  
- Diffusion model training on large wound datasets  
- Fine-tuning on limited leprosy imagery  

## Repository Structure

- `preprocessing/` — image and data processing 
- `notebooks/` — Experimentation and training notebooks  
- `samples/` — Sample outputs  
