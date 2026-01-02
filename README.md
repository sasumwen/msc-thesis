# MSc Thesis – Boundary-Refined Segmentation and Multimodal Learning for Brain MRI Analysis

This repository contains selected code, scripts, and experimental artifacts developed as part of my MSc (Computer Science) thesis at the University of Lethbridge.

**Thesis Title**  
Advanced Boundary-Enhanced Instance Segmentation and Spatial-Temporal Transformer Models for Automated Schizophrenic Investigation

---

## Why this work matters

Accurate boundary delineation and multimodal fusion remain major challenges in medical imaging systems deployed beyond research settings. This project focuses on closing that gap by combining boundary-aware segmentation, transformer-based modeling, and performance-conscious engineering practices.

The work emphasizes:
- Translating research ideas into reproducible pipelines
- Designing models that scale on real GPU infrastructure
- Treating preprocessing, training, and evaluation as a single system

---

## Technical Contributions

### 1. BoRefAttnNet – Boundary-Refined 3D Segmentation
- Boundary-aware 3D U-Net variant with multi-scale attention
- Explicit focus on anatomical boundary precision
- Evaluated using Dice, Hausdorff Distance (HD), and Average Surface Distance (ASD)
- Demonstrated measurable improvements over standard 3D U-Net baselines

### 2. DySTTM – Dynamic Spatial-Temporal Transformer Model
- Fuses structural MRI segmentation outputs with functional MRI time-series
- Models spatial–temporal dependencies for schizophrenia-related pattern detection
- Achieved improved ROC-AUC over 3D CNN baselines

---

## Repository Structure


---

## Data and Preprocessing

- Structural MRI preprocessing:
  - AFNI (skull stripping, alignment)
  - FastSurfer (cortical parcellation and labeling)
- Dataset:
  - COBRE (accessed via SchizConnect)
- Raw MRI data is **not included** due to privacy, ethics, and licensing constraints

All pipelines were designed to be deterministic, GPU-compatible, and reproducible.

---

## Training, Evaluation, and Engineering

- Frameworks: PyTorch, TensorFlow
- Training strategy:
  - Patch-based 3D training for memory efficiency
  - Boundary-aware loss formulations
- Metrics:
  - Dice, HD, ASD (segmentation)
  - ROC-AUC (classification)
- Experiments organized to support iteration, ablation, and reproducibility

---

## High-Performance Computing Context

This work was informed by formal HPC training:

- Selected as **one of 10 Canadian researchers** by the Digital Research Alliance of Canada
- Attended the **2025 International High-Performance Computing Summer School (IHPCSS)** in Lisbon
- Focus areas:
  - GPU acceleration
  - Parallel workflows
  - Performance-aware model design

This directly influenced model design choices and training efficiency.

---

## Research to Production Notes

This repository reflects the **research backbone** of the thesis. Extended experiments, infrastructure automation, and deployment-oriented components live in private repositories due to compute cost, IP considerations, and ongoing development.

What I can walk through in detail:
- Architecture and loss design decisions
- Failure cases and boundary error modes
- Scaling trade-offs and GPU constraints
- How these models translate to production ML systems

---

## Author

**Osasumwen Raphael Imarhiagbe (Osasu)**  
MSc Computer Science – University of Lethbridge  
Machine Learning Engineer | Applied AI Researcher  


