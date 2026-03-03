# 🛡️ Credit Card Fraud Detection Using Ensemble Learning & Deep Learning
AN PROJECT BY  RAGUL RAVICHANDIRAN

A comprehensive final-year project that replicates existing ensemble-learning research **and** proposes novel deep-learning and explainability enhancements for identifying fraudulent credit-card transactions.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Project Structure](#project-structure)  
3. [Datasets](#datasets)  
4. [Installation](#installation)  
5. [Quick Start](#quick-start)  
6. [Reproducing Results](#reproducing-results)  
7. [Dashboard](#dashboard)  
8. [Notebooks](#notebooks)  
9. [Models Implemented](#models-implemented)  
10. [Results Highlights](#results-highlights)  
11. [Hardware / Software Requirements](#hardware--software-requirements)  
12. [License](#license)

---

## Project Overview

| Aspect | Details |
|--------|---------|
| **Goal** | Detect fraudulent credit-card transactions with high recall while minimising false positives |
| **Existing Work** | Gaussian Naive Bayes, Random Forest, XGBoost on two datasets with oversampling / undersampling / SMOTE |
| **Proposed Work** | CNN-BiGRU, DistilBERT, Stacking Ensemble, SHAP/LIME explainability, 2FA/MFA simulation |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC, paired t-test, McNemar's test, Friedman-Nemenyi |

---

## Project Structure

```
credit-card-fraud-detection/
├── data/raw/                   # Raw CSVs (user-supplied)
├── data/processed/             # Preprocessed splits
├── models_saved/               # Trained model files
├── notebooks/
│   ├── 01_eda_analysis.ipynb
│   ├── 02_existing_work_replication.ipynb
│   ├── 03_proposed_model_development.ipynb
│   └── 04_model_comparison_analysis.ipynb
├── src/
│   ├── data/                   # Preprocessing & balancing
│   ├── models/                 # All model definitions
│   ├── visualization/          # Plotting utilities
│   └── utils/                  # Config, metrics, helpers
├── dashboard/app.py            # Streamlit dashboard
├── reports/figures/            # Generated plots
├── reports/final_report/       # Technical report
├── main.py                     # CLI entry point
├── requirements.txt
├── setup.py
└── README.md
```

---

## Datasets

### 1. European Cardholders (Real-world)
- **Source**: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions · 492 fraudulent (0.172%)
- 28 PCA-transformed features + Time + Amount

### 2. Sparkov Simulated
- **Source**: [Kaggle — Fraud Detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- ~1.85M simulated transactions with demographics, merchant info, temporal data
OPTIONAL TO USE SINGLE OR MULTIPLE DATASET , 
TO TRAIN AND TEST USE MULTIPLE DATASET MODEL IF YOU WANT ONLY TRAINING THEN USE SINGLE MODEL 

### Download Instructions

1. **With Kaggle CLI** (recommended):
   ```bash
   pip install kaggle
   # Place kaggle.json in ~/.kaggle/
   kaggle datasets download mlg-ulb/creditcardfraud -p data/raw/ --unzip
   kaggle datasets download kartik2112/fraud-detection -p data/raw/ --unzip
   ```
EXTRACT IT AND PLACE INTO THE DATA FOLDER AS PER RUNNING RECOMMENDATION 
2. **Manual**: Visit the links above, download, and extract CSVs into `data/raw/`.

---

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd credit-card-fraud-detection

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install project as editable package
pip install -e .
```

---

## Quick Start

```bash
# 1. Preprocess datasets (after downloading CSVs)
python main.py preprocess

# 2. Train paper-replication models (NB, RF, XGB)
python main.py train --existing

# 3. Train proposed models (CNN-BiGRU, BERT, Stacking + tuning)
python main.py train --proposed

# 4. Generate evaluation plots
python main.py evaluate

# 5. Run explainability (SHAP, LIME)
python main.py explain

# 6. Launch interactive dashboard
python main.py dashboard

# Or run the full pipeline end-to-end:
python main.py all
```

---

## Reproducing Results

1. Download both datasets into `data/raw/`.
2. Run `python main.py all`.
3. Results CSV → `models_saved/existing_results.csv`.
4. Figures → `reports/figures/`.
5. Open notebooks in Jupyter for cell-by-cell analysis.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

**Pages:**
- 📊 **Data Overview** — dataset stats, distributions, heatmaps
- 📈 **Model Performance** — interactive bar charts, heatmap comparisons
- 🔮 **Prediction Interface** — input transaction features, get prediction
- 🔍 **Explainability** — SHAP/LIME plot viewer
- 🔐 **2FA Simulation** — risk-based multi-factor auth demo
- 📅 **Time Series** — fraud trends by hour, amount analysis

---

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_eda_analysis.ipynb` | EDA: class imbalance, feature distributions, correlations |
| 02 | `02_existing_work_replication.ipynb` | Paper replication: NB / RF / XGB × 4 strategies × 2 datasets |
| 03 | `03_proposed_model_development.ipynb` | CNN-BiGRU, BERT, Stacking, tuning, SHAP/LIME, 2FA |
| 04 | `04_model_comparison_analysis.ipynb` | Statistical tests, time benchmarks, final comparison |

---

## Models Implemented

### Existing Work (Paper Replication)
| Model | Type | Parameters |
|-------|------|------------|
| Gaussian Naive Bayes | Baseline | Default |
| Random Forest | Bagging | n_estimators=5, max_depth=3 |
| XGBoost | Boosting | n_estimators=5, max_depth=3 |

### Proposed Enhancements
| Model | Type | Description |
|-------|------|-------------|
| CNN-BiGRU | Deep Learning | Conv1D → BiGRU → Dense |
| DistilBERT | Transformer | Tabular→text, fine-tuned |
| Stacking Ensemble | Meta-learning | RF + XGB → Logistic Regression |
| Tuned RF | Bagging | RandomizedSearchCV optimised |
| Tuned XGB | Boosting | RandomizedSearchCV optimised |

---

## Results Highlights

- **XGBoost + SMOTE** consistently achieves the best F1 and AUC among existing models.
- **Stacking Ensemble** and **Tuned XGB** further improve performance.
- **CNN-BiGRU** is competitive for the European dataset.
- **SMOTE** outperforms random oversampling and undersampling across all models.
- Statistical tests confirm significant differences between ensemble and baseline methods.

---

## Hardware / Software Requirements

| Requirement | Minimum |
|-------------|---------|
| Python | 3.8+ |
| RAM | 16 GB |
| Storage | 50 GB free |
| GPU | Recommended for deep learning |
| OS | Windows / Linux / macOS |

---

## License
RAGUL RAVICHANDIRAN
FINAL YEAR PROJECT (DEPARTMENT OF AI&DS)
This project is for academic / educational use.
