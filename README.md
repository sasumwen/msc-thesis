# MSc Thesis: Boundary-Refined Segmentation and Multimodal Learning for Schizophrenia Analysis

This repository contains code, scripts, and experimental artifacts developed as part of my Master of Science (Computer Science) thesis at the University of Lethbridge.

**Thesis Title**  
Advanced Boundary-Enhanced Instance Segmentation and Spatial-Temporal Transformer Models for Automated Schizophrenic Investigation

---

## Overview

This work focuses on developing robust, boundary-aware deep learning models for brain MRI analysis and extending them into multimodal learning systems for schizophrenia research. The project combines classical neuroimaging pipelines with modern deep learning architectures, emphasizing reproducibility, performance, and research-to-production workflows.

Key contributions include:
- Boundary-refined 3D segmentation networks for structural MRI
- Multimodal fusion of structural and functional MRI
- End-to-end preprocessing, training, evaluation, and logging pipelines
- Performance-aware experimentation on GPU infrastructure

---

## Core Contributions

### 1. BoRefAttnNet (Boundary-Refined Attention Network)
- A boundary-aware 3D U-Net variant with multi-scale attention
- Designed to improve anatomical boundary delineation in brain MRI
- Evaluated using Dice, Hausdorff Distance (HD), and Average Surface Distance (ASD)
- Demonstrated significant improvements over standard 3D U-Net baselines

### 2. DySTTM (Dynamic Spatial-Temporal Transformer Model)
- Integrates structural MRI segmentation outputs with functional MRI time-series
- Models spatial-temporal dependencies for schizophrenia detection
- Achieved improved ROC-AUC over 3D CNN baselines

---

## Repository Structure

