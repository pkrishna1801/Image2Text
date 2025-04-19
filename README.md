

# 🩺 U-Xray: Chest X-ray Report Generation

## 📚 Dataset Information

The **IU-XRay** (Indiana University Chest X-Rays) dataset contains a collection of chest X-ray images and their corresponding diagnostic reports, collected by Indiana University. It includes **7,470** frontal or lateral chest X-ray images and **3,955** corresponding reports.

Each radiology report includes four sections:
- **Comparison**: Prior patient history (e.g., previous exams)
- **Indication**: Symptoms or reasons for the exam (e.g., hypoxia)
- **Findings**: Radiological observations
- **Impression**: Final diagnosis

A robust system should ideally generate both the **Findings** and **Impression** sections, potentially connecting them for coherent medical summaries.

---

## 🏥 Importance

Radiologists and physicians interpret large volumes of biomedical images (X-rays, CT, PET scans) daily. Automating report generation helps:
- Reduce diagnostic errors
- Support less experienced doctors with suggestions
- Lower the cost per patient examination

The IU-XRay dataset was an early contribution to this domain and continues to be a benchmark for evaluating large multimodal medical models (e.g., MedDr, RadFM).

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
