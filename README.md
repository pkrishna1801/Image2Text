
# 📄 Chest X-ray Dataset Preprocessing

This project demonstrates how to load and preprocess the [Indiana University Chest X-ray dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) using Python and PyTorch. It includes utilities to download the dataset via `kagglehub`, merge metadata, and begin dataset preparation for machine learning.

---

## 📁 Dataset

- **Source**: [Kaggle - Indiana University Chest X-rays](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- **Contents**:
  - X-ray image files
  - `indiana_projections.csv`: Projection metadata
  - `indiana_reports.csv`: Radiology reports and study identifiers

---

## ⚙️ Setup

Install required libraries:

```bash
pip install kagglehub pandas numpy pillow torch
```

---

## 📦 Code Overview

### 1. Download Dataset

```python
import kagglehub
path = kagglehub.dataset_download("raddar/chest-xrays-indiana-university")
```

### 2. Load Metadata

```python
import pandas as pd

projections_df = pd.read_csv('.../indiana_projections.csv')
reports_df = pd.read_csv('.../indiana_reports.csv')
merged_df = pd.merge(projections_df, reports_df, on='uid')
```

### 3. Prepare Dataset Class (Partially Implemented)

A custom PyTorch `Dataset` class is scaffolded for future use with Dataloaders:

```python
from torch.utils.data import Dataset

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        ...
    
    def __len__(self):
        ...
    
    def __getitem__(self, idx):
        ...
```

---

## 🧠 Future Work

- Complete the `ChestXrayDataset` class
- Add transformations (e.g., `torchvision.transforms`)
- Visualize data samples
- Fine-tune or train a vision-language model for report generation

---

## 📌 Notes

- Ensure the `kaggle.json` API key is configured for `kagglehub` to download datasets.
- File paths in the notebook assume a Kaggle environment; update accordingly for local use.
