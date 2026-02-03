# From-Pseudo-Segmentation-to-Transformer-Based-3D-Brain-Tumor-Segmentation.

This repository will contain code files about the exploratory study performed to bridge pseudo segmentation with transformer based 3D segmentation. The information regarding running the code files can be found in README file available in this repository. Drive link to access the model which was applied on BraTS data is present in this readme section along with both the datasets(2D and 3D) which were used for this exploratory work. A research paper regarding the same has been submitted as e-print in the TechrXiv. The study focuses on methodological validation, loss design, and optimization strategies for medical image segmentation under limited annotation availability.

# Dataset and Model link 

Dataset for pseudo segmentation : https://www.kaggle.com/datasets/fernando2rad/brain-tumor-mri-images-44c

BraTS Preliminary release 2025 for Transformer Based 3D segmentation : https://www.kaggle.com/datasets/patiencebwire/brats-2025

Final Model applied on the BraTS dataset which gave the best results and was documented in the paper : https://drive.google.com/drive/folders/1GBegUQp5DfBvFFCFNJfSPT6BHJ9PsPoS?usp=sharing

# Project Overview

Due to high cost and limited availability of 3D annotated voxel level dataset, we have established a step-by-step pipeline from pseudo segmentation on dataset consisiting of 44 brain tumor classes and implementing UNET-R model on it. We then switched to preliminary release of BraTS data 2025. We worked on it without performing any pseudo segmentation and got excellent results. Along the way, various parameters were tuned and the best model was saved. 
The project has aimed to achieve the following :-

  (1) Validating segmentation pipelines under weak supervision using pseudo masks.

  (2) Transitioning to fully supervised volumetric learning using a transformer-based UNETR model.

  (3) Systematically optimizing loss functions and training hyperparameters to achieve robust generalization.

# Methodology 

## To run the files / codes / models provided 

I cannot provide datasets in the repository because of legal constraints. I have provided link to datasets here in the README file. I have also created a complete folder structure for you to operate with. If you are trying to run these code files in your local system then make sure dataset is arranged in the format give. You can only get errors in the code if file address is not matching with your system, rest everything in the code is correct. For your help, even markdowns are also given in the coding files which are there to explain important steps and provide detailed conceptual explanations.

Also to add in the same line, BraTS dataset will automatically come downloaded in the format as shown in repository for the 2D dataset you would need to arrange it in the format given, for which you can simply run the file To_split_dataset.ipynb but make sure to match file addresses and names.

## Pseudo-Segmentation (Weak Supervision)

(1) In the initial stage, voxel-level tumor annotations were not available. To validate the segmentation pipeline, pseudo masks were generated using classical image processing techniques.

(2) A public brain MRI dataset containing 44 tumor classes was used.

(3) MRI slices were converted to grayscale and resized to a fixed resolution.

(4) Otsu thresholding was applied to separate foreground and background regions.

(5) Morphological operations were used to remove noise and fill gaps.

(6) The resulting masks were binarized to obtain pseudo tumor regions.

(7) These pseudo masks do not represent clinically accurate tumor boundaries but serve as weak supervisory signals for validating data loading, preprocessing, loss functions, and evaluation metrics.

## Baseline 2D Segmentation

Using the pseudo-labeled data, a 2D U-Net model was trained to perform binary segmentation.

Input: single-channel MRI slices

Output: binary tumor mask

Loss function: Dice loss

The model was trained for 20 epochs, and performance was evaluated using Dice score and IoU. High internal consistency scores confirmed the correctness of the segmentation pipeline and metric implementation before transitioning to volumetric learning.

## Transformer-Based 3D Segmentation

For volumetric segmentation, a UNETR-based architecture was used.

A Vision Transformer encoder processes non-overlapping 3D patches.

Transformer features are passed to a U-Net–style decoder via skip connections.

The decoder progressively upsamples features to recover full spatial resolution.

The final output is a single-channel volumetric tumor probability map.

## Loss Function Design

To address severe foreground–background imbalance and stabilize training, a composite loss function was used:

Dice loss to optimize region-level overlap.

Binary cross-entropy (BCE) to provide voxel-wise supervision.

The two losses were combined using a weighting factor, which was tuned empirically using validation performance.

## Training and Hyperparameter Tuning

Optimizer: AdamW

Batch size (3D): 1

Epochs: 20

Learning rate and BCE–Dice loss weighting were systematically tuned using short validation runs. The final model was trained using the best-performing hyperparameters selected from these experiments.

## Evaluation and Sanity Checks

Segmentation performance was primarily evaluated using the Dice score, with IoU as a supplementary metric.

To verify evaluation correctness, a sanity check was performed by computing Dice score for an all-background prediction. The near-zero Dice value confirmed that the evaluation pipeline was correctly implemented and not biased by class imbalance.

## Why This Approach

Brain tumor segmentation presents multiple practical challenges, including limited availability of voxel-level annotations, severe foreground–background imbalance, and high computational cost associated with volumetric models. Training transformer-based 3D segmentation architectures directly on limited annotated data can be inefficient and error-prone if the data pipeline, loss formulation, or evaluation metrics are incorrectly implemented.

This work adopts a progressive weak-to-strong supervision strategy to mitigate these challenges. The pseudo-segmentation stage allows early validation of data loading, preprocessing, loss functions, and evaluation metrics using inexpensive 2D models and weak supervisory signals. By confirming pipeline correctness and training stability at this stage, the risk of silent failures during computationally expensive 3D training is significantly reduced.

The transition to fully supervised 3D segmentation using a UNETR-based architecture leverages the strengths of transformer models for capturing global context in volumetric data while preserving spatial detail through U-Net–style skip connections. Formulating the task as binary segmentation further reduces complexity and class imbalance, enabling more stable optimization under limited data conditions.

Overall, this staged approach emphasizes methodological reliability and reproducibility over premature optimization, ensuring that architectural choices, loss design, and evaluation metrics are well understood before scaling to full volumetric learning. This strategy is particularly suitable for low-data medical imaging scenarios and research-oriented experimentation.

# Results

The proposed weak-to-strong supervision pipeline was evaluated across both the pseudo-segmentation stage and the fully supervised 3D segmentation stage. Performance was measured using overlap-based metrics, primarily the Dice score, with IoU used as a supplementary metric.

## Pseudo-Segmentation (2D)

Purpose: Validate data handling, loss formulation, and metric correctness under weak supervision.

Model: 2D U-Net trained on pseudo-labeled MRI slices.

Performance:

Mean Dice: 0.97

Mean IoU: 0.94

Interpretation: These scores indicate high internal consistency and confirm the correctness of the segmentation pipeline. As pseudo masks are not clinically accurate, these results are treated as methodological validation rather than medical performance.

## Fully Supervised 3D Segmentation

Dataset: Preliminary BraTS 2025 release (88 volumes; train/val/test split).

Model: UNETR (Vision Transformer encoder with U-Net–style decoder).

Task: Binary tumor segmentation (all tumor classes merged).

Training:

Optimizer: AdamW

Loss: Combined Dice + BCE

## Hyperparameters tuned using validation performance.

Performance:

Baseline Validation Dice: 0.75

After Hyperparameter Tuning:

Validation Dice: 0.80

Test Dice: 0.81

## Evaluation Sanity Check

To verify the correctness of the evaluation pipeline, a sanity check using an all-background prediction was performed. This produced a near-zero Dice score, confirming that the reported results reflect genuine tumor localization rather than metric artifacts or class imbalance bias. 


