# llm-judge-powerline
# LLM-as-Judge for Semantic Judging of Powerline Segmentation in UAV Inspection

This repository contains the code and experiment pipeline used in the paper **“LLM-as-Judge for Semantic Judging of Powerline Segmentation in UAV Inspection”**. The goal of the project is to evaluate whether a multimodal large language model can act as a semantic judge for segmentation quality when ground-truth labels are unavailable at deployment time.

The overall workflow is:
1. Train a **U-Net** segmentation model with a **ResNet34 backbone** on the **TTPLA** dataset.
2. Train the model on clean TTPLA train images.
3. Create a **challenge set** from the TTPLA test split by adding controlled visual corruptions.
4. Run the trained U-Net on the corrupted images.
5. Overlay the predicted segmentation masks on the RGB images using a **red mask overlay**.
6. Send the overlay images to an **LLM-as-Judge** with a fixed prompt.
7. Measure:
   - **Repeatability / stability** of the judge under repeated identical inputs.
   - **Sensitivity** of the judge under controlled visual corruptions.

---

## 1. Project objective

The purpose of this project is **not** to improve the segmentation network itself. Instead, it studies whether an LLM can reliably evaluate the quality of powerline segmentation overlays and behave consistently enough to serve as an offboard semantic watchdog for UAV inspection.

The LLM judge outputs three values for each overlay image:

- `score`: discrete segmentation quality score in **{1, 2, 3, 4, 5}`
- `confidence`: scalar confidence in **[0, 1]`
- `explanation`: short textual rationale

The experiments in the paper are organized into two sections:

- **Section A: Repeatability / Stability**
- **Section B: Sensitivity under controlled corruptions**

---

## 2. Dataset

### 2.1 Primary dataset

This work uses the **TTPLA** aerial powerline dataset.

- Dataset: **TTPLA: An aerial-image dataset for detection and segmentation of transmission towers and power lines**
- Use the **official training split** to train the segmentation model.
- Use the **official test split** for evaluation and to build the challenge set.

### 2.2 Data used in this paper

Two evaluation sets are used:

1. **Clean test overlays**
   - The original TTPLA test images.
   - Used for training the base UNet model.

2. **Challenge set**
   - Built from the TTPLA test images.
   - Synthetic corruptions are added to simulate adverse operating conditions.
   - The corrupted images are then segmented by the trained U-Net.
   - The predicted masks are overlaid on the corrupted RGB images and judged by the LLM.

### 2.3 Corruption types used in the challenge set

The challenge set includes the following corruption families:

- `fog`
- `rain`
- `snow`
- `shadow`
- `sunflare`

Each corruption is applied at **three severity levels**:

- severity 1
- severity 2
- severity 3

In the paper, the challenge-set corruptions were generated using **Albumentations**.

---

## 3. Preprocessing and overlay generation

### 3.1 Segmentation preprocessing

The segmentation model is trained on TTPLA images and labels using the preprocessing defined in the training code.

### 3.2 Challenge-set generation

Starting from the TTPLA **test images**, controlled corruptions are added to generate the challenge set. The corrupted RGB image is then passed through the trained U-Net to obtain a predicted mask.

### 3.3 Overlay generation

For both clean and corrupted test images:

1. Run the trained U-Net.
2. Save the predicted segmentation mask.
3. Overlay the predicted mask on the RGB image.
4. Use a **red semi-transparent mask overlay** as the visual input to the LLM judge.

These overlay images are the actual inputs used by the judge scripts.

---

## 4. Models used

### 4.1 Segmentation model

- **Architecture:** U-Net
- **Encoder / backbone:** ResNet34
- **Training epochs:** 25
- **Learning rate:** `1e-3`

### 4.2 LLM-as-Judge model

The paper uses **GPT-4o** as the main multimodal judge.

The judging scripts also support:

- `openai` backend
- `gemini` backend
- `mock` backend

The repeatability script explicitly supports `gpt-4o`, `gemini-1.5-pro`

## 5. Training / fine-tuning procedure

### 5.1 Segmentation model training

Train a U-Net with a ResNet34 backbone on the TTPLA training split.

**Training configuration used in the paper:**

- model: `U-Net`
- encoder: `ResNet34`
- epochs: `25`
- learning rate: `1e-3`

### 5.2 Segmentation inference

After training, run inference on:

- the **clean TTPLA test split**
- the **corrupted challenge-set images**

### 5.3 Challenge-set generation

Use the TTPLA test images to create the corrupted evaluation set.

## 6. Environment setup and scripts to run

Create a Python environment and install the required packages.
```
pip install numpy pandas scipy matplotlib pillow openai google-generativeai albumentations
```
### Run exp1_repeatability_5runs.py on the overlay images to execute the LLM judge five times under identical settings and generate the repeatability metrics.

### Run exp3_judge_and_aggregate.py on the corrupted challenge-set overlays to produce condition-wise CSV outputs, then run analyze_sensitivity_sectionB.py to compute the sensitivity statistics used for Table II and the corresponding robustness plots across corruption types and severity levels.
