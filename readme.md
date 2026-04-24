# ImmuCAGMoE: Cross-Attention Guided Mixture-of-Experts for Antigen Binding Prediction

## Overview

This repository contains the implementation code for the paper "ImmuCAGMoE: a cross-attention guided mixture-of-experts architecture with multiple embedding adaptation for predicting antigen binding to HLAs and TCRs". ImmuCAGMoE is an immunology-oriented framework that combines Multi-Embedding Adaptation (MEA) and Cross-Attention Guided Mixture-of-Experts (CAGMoE) for accurate prediction of antigen binding to HLAs and TCRs.

## Environment Setup

### Prerequisites
- Python 3.12 (recommended)
- pip or conda

### Installation Steps

1. Create a new Python environment:
   ```bash
   conda create -n immucagmoe python=3.12
   conda activate immucagmoe
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   where requirements.txt is in the [root directory]/libs/.

## Directory Structure

```
ImmuCAGMoE/
├── 01_ESMcEmbeddingGeneration.py  # Generate ESM embeddings for protein sequences
├── 02_5_fold_evaluation_pHLA.py   # 5-fold cross-validation for antigen-HLA binding prediction
├── 02_5_fold_evaluation_pTCR.py   # 5-fold cross-validation for antigen-TCR binding prediction
├── 03_train_pHLA.py               # Train model for antigen-HLA binding prediction
├── 03_train_pTCR.py               # Train model for antigen-TCR binding prediction
├── 04_evaluation_pHLA.py          # Evaluate model on all antigen-HLA datasets
├── 04_evaluation_pTCR.py          # Evaluate model on all antigen-TCR datasets
├── configs/                        # Configuration files
│   ├── ESMcEmbedding.yaml
│   ├── config_5_fold_evaluation_pHLA.yaml
│   ├── config_5_fold_evaluation_pTCR.yaml
│   ├── config_evaluation_pHLA.yaml
│   ├── config_evaluation_pTCR.yaml
│   ├── config_model.yaml
│   ├── config_train_pHLA.yaml
│   └── config_train_pTCR.yaml
├── data/                           # Dataset files
│   ├── data_pHLA/                  # antigen-HLA binding datasets
│   │   ├── HPV_set.csv
│   │   ├── dataset.csv
│   │   ├── external_set.csv
│   │   ├── independent_set.csv
│   │   ├── neoantigen_set.csv
│   │   ├── train_fold_1.csv
│   │   ├── train_fold_2.csv
│   │   ├── train_fold_3.csv
│   │   ├── train_fold_4.csv
│   │   ├── train_fold_5.csv
│   │   ├── val_fold_1.csv
│   │   ├── val_fold_2.csv
│   │   ├── val_fold_3.csv
│   │   ├── val_fold_4.csv
│   │   └── val_fold_5.csv
│   └── data_pTCR/                  # antigen-TCR binding datasets
│       ├── COVID_set.csv
│       ├── dataset.csv
│       ├── external_set.csv
│       ├── independent_set.csv
│       ├── train_fold_1.csv
│       ├── train_fold_2.csv
│       ├── train_fold_3.csv
│       ├── train_fold_4.csv
│       ├── train_fold_5.csv
│       ├── val_fold_1.csv
│       ├── val_fold_2.csv
│       ├── val_fold_3.csv
│       ├── val_fold_4.csv
│       └── val_fold_5.csv
├── libs/                           # Custom libraries
│   ├── AtchleyFactors/             # Physicochemical properties for protein sequence embedding
│   │   └── Atchley_factors.csv
│   ├── ESMc/                       # ESM model and embedding files for protein sequence embedding
│   │   └── ESMcEmbedding/          # Generated embeddings by ESM model
│   │   └── modelStructureState/    # ESM model structure and state
│   │       └── esm300m_structure_and_state
│   ├── model_states/               # Recommended model states of the ImmuCAGMoE model 
│   │   ├── pHLA_model_final.pth
│   │   └── pTCR_model_final.pth
│   ├── utils/                      # Utility functions
│   │   ├── ESMcEmbedding.py
│   │   ├── embedding.py
│   │   ├── optimization.py
│   │   └── utils.py
│   ├── dataset.py                  # Dataset class definitions
│   ├── models.py                   # Model architecture definitions
│   └── requirements.txt            # Additional dependencies
├── results/                        # Output directory
│   ├── 5_fold_evaluation/          # 5-fold cross-validation results
│   │   ├── pHLA/
│   │   │   ├── logs/
│   │   │   └── model_states/
│   │   └── pTCR/
│   │       ├── logs/
│   │       └── model_states/
│   ├── evaluation/                 # Final evaluation results
│   │   ├── pHLA/
│   │   └── pTCR/
│   └── train/                      # Training results
│       ├── pHLA/
│       │   ├── logs/
│       │   └── model_states/
│       └── pTCR/
│           ├── logs/
│           └── model_states/
└── README.md                       # This file
```

## Usage Instructions

If you do not need to reimplement the 5-fold cross-validation or training, you can skip the step 2 and 3, and follow the step 1 and 4 directly.

### 1. Generate ESM Embeddings

Before running any training or evaluation scripts, you need to generate ESM embeddings for protein sequences:

```bash
python 01_ESMcEmbeddingGeneration.py
```

This script will:
- Load configuration from `configs/ESMcEmbedding.yaml`
- Generate embeddings for sequences in the specified data directories
- Save embeddings to the specified output directory
- Please make sure to configure all the settings in the **configs/ESMcEmbedding.yaml**

### 2. Run 5-Fold Cross-Validation (If needed)


To perform 5-fold cross-validation for antigen-HLA binding prediction:

```bash
python 02_5_fold_evaluation_pHLA.py
```
- Before running the 5-fold cross-validation scripts, please make sure to configure all the settings in the **configs/config_5_fold_evaluation_pHLA.yaml**
  

To perform 5-fold cross-validation for antigen-TCR binding prediction:

```bash
python 02_5_fold_evaluation_pTCR.py
```
- Before running the 5-fold cross-validation scripts, please make sure to configure all the settings in the **configs/config_5_fold_evaluation_pTCR.yaml**

### 3. Train Models (If needed)

To train the model for antigen-HLA binding prediction:

```bash
python 03_train_pHLA.py
```
- Before running the training scripts, please make sure to configure all the settings in the **configs/config_train_pHLA.yaml**


To train the model for antigen-TCR binding prediction:

```bash
python 03_train_pTCR.py
```
- Before running the training scripts, please make sure to configure all the settings in the **configs/config_train_pTCR.yaml**

### 4. Evaluate Models

To evaluate the model on all antigen-HLA datasets:

```bash
python 04_evaluation_pHLA.py
```
- Before running the evaluation scripts, please make sure to configure all the settings in the **configs/config_evaluation_pHLA.yaml**


To evaluate the model on all antigen-TCR datasets:

```bash
python 04_evaluation_pTCR.py
```
- Before running the evaluation scripts, please make sure to configure all the settings in the **configs/config_evaluation_pTCR.yaml**

## Output

All execution results, including model checkpoints, performance metrics, and logs, will be saved in the `results/` directory with the following structure:

- `results/5_fold_evaluation/`: Results from 5-fold cross-validation
- `results/evaluation/`: Final evaluation results on all datasets
- `results/train/`: Training logs and model checkpoints

## Citation

The paper has not peen published yet.
We will give the citation once the paper is published.
<!-- 
If you use this code in your research, please cite the following paper:

```
@article{ImmuCAGMoE2026,
  title={ImmuCAGMoE: a cross-attention guided mixture-of-experts architecture with multiple embedding adaptation for predicting antigen binding to HLAs and TCRs},
  author={[Authors]},
  journal={[Journal]},
  year={2026},
  volume={[Volume]},
  pages={[Pages]}
}
``` -->

## Contact

For any questions or issues, please contact the corresponding author of the paper.