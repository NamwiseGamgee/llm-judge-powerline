# llm-judge-powerline
# LLM-as-Judge for Semantic Judging of Powerline Segmentation in UAV Inspection

This repository contains the code and experiment pipeline for **“LLM-as-Judge for Semantic Judging of Powerline Segmentation in UAV Inspection.”** The project evaluates whether a multimodal LLM can act as a semantic judge for powerline segmentation quality when ground-truth labels are unavailable at deployment time.

## Overview

The experimental pipeline consists of four stages:

1. Train a **U-Net with a ResNet34 backbone** on the **TTPLA** training split.
2. Build a **challenge set** from the **TTPLA test split** by applying controlled corruptions (`fog`, `rain`, `snow`, `shadow`, `sunflare`) at three severity levels.
3. Get inference from the trained U-Net on corrupted test images, then create **red semi-transparent segmentation overlays**.
4. Evaluate the overlays with an **LLM-as-Judge** using a fixed prompt and analyze:
   - **Repeatability / stability**
   - **Sensitivity under controlled corruptions**

## Dataset

This work uses the **TTPLA** aerial powerline dataset.

- **Training split:** used to train the segmentation model
- **Test split:** used for clean evaluation and challenge-set construction

The challenge set is created from the TTPLA test images by applying synthetic corruptions with **Albumentations**.

## Models and Configuration

### Segmentation model
- **Architecture:** U-Net
- **Backbone:** ResNet34
- **Epochs:** 25
- **Learning rate:** `1e-3`

### LLM-as-Judge
- **Primary model:** GPT-4o

The judging scripts also support:
- `openai`
- `gemini`
- `mock`

## Preprocessing and Overlay Generation

For both clean and corrupted test images:

1. Run the trained U-Net to generate predicted segmentation masks.
2. Overlay the predicted masks on the RGB images using a **red semi-transparent mask**.
3. Use the overlay images as input to the LLM judge.

## Experiments

### Section A — Repeatability / Stability
Run `exp1_repeatability_5runs.py` on the overlay images to execute the LLM judge five times under identical settings and generate the repeatability metrics reported in Tables I(a–f), including score agreement, confidence stability, ICC(1,1), pairwise correlations, and text-overlap statistics.

### Section B — Sensitivity
Run `exp3_judge_and_aggregate.py` on the corrupted challenge-set overlays to generate condition-wise CSV outputs, then run `analyze_sensitivity_sectionB.py` to compute the sensitivity statistics used for Table II and the corresponding robustness plots across corruption types and severity levels.

## Environment Setup

Install the required packages:

```bash
pip install numpy pandas scipy matplotlib pillow openai google-generativeai albumentations
