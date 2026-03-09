# Telco Customer Churn Prediction
### Feature Engineering Pipeline with Scikit-learn & TabNet Deep Learning Model

---

## Project Overview

This project is part of a hands-on ML engineering task focused on building a robust feature engineering pipeline and training a deep learning model to predict customer churn for a telecommunications company.

The goal is to identify customers likely to churn (cancel their subscription) based on their demographics, service usage, and billing information — enabling the business to take proactive retention actions.

The pipeline covers end-to-end steps: data cleaning, exploratory data analysis (EDA), feature engineering, preprocessing with Scikit-learn, model training using TabNet, and comprehensive evaluation using classification metrics including ROC-AUC, Confusion Matrix, Precision-Recall Curve, and SHAP values.

---

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 rows × 21 columns
- **Target Variable:** `Churn` (Yes / No → encoded as 1 / 0)
- **Class Distribution:** ~73% No Churn (5,174) / ~27% Churn (1,869) — imbalanced

### Feature Types

| Type | Features |
|------|----------|
| Numerical | `tenure`, `MonthlyCharges`, `TotalCharges` |
| Categorical | `Contract`, `InternetService`, `PaymentMethod`, `gender`, etc. |
| Binary | `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, etc. |
| Add-on Services | `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |

---

## Project Structure

```
telco-churn-prediction/
│
├── telco_churn_pipeline.py        # Data cleaning, EDA, feature engineering & preprocessing
├── telco_churn_tabnet.py          # TabNet model training & evaluation
├── Telco-Customer-Churn.csv       # Raw dataset
├── telco_churn_preprocessed.csv   # Preprocessed data (SageMaker-ready, no header, target first)
├── README.md                      # Project documentation
│
├── outputs/
│   ├── churn_distribution.png
│   ├── numerical_distributions.png
│   ├── boxplots_outliers.png
│   ├── categorical_churn_rates.png
│   ├── correlation_heatmap.png
│   ├── tenure_vs_charges.png
│   ├── tabnet_confusion_matrix.png
│   ├── tabnet_roc_curve.png
│   ├── tabnet_pr_curve.png
│   ├── tabnet_training_history.png
│   ├── tabnet_feature_importance.png
│   ├── shap_churn.png
│   ├── shap_no_churn.png
│   └── shap_dot_churn.png
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install pytorch-tabnet
pip install shap
pip install boto3          # for S3 upload to SageMaker
```

---

## Pipeline & Feature Engineering

### 1. Data Cleaning
- Dropped `customerID` (irrelevant identifier)
- Fixed `TotalCharges` — stored as object due to whitespace; converted to numeric
- Imputed 11 missing `TotalCharges` values with median
- Encoded target `Churn` as binary (Yes→1, No→0)

### 2. Exploratory Data Analysis
Six visualizations were produced:
- Churn distribution (pie + count plot)
- Numerical feature histograms split by churn
- Boxplots for outlier detection (tenure, MonthlyCharges, TotalCharges)
- Churn rate bar charts for all 16 categorical features
- Correlation heatmap
- Tenure vs Monthly Charges scatter plot

**Key EDA Findings:**
- `tenure` has the strongest correlation with churn (-0.35) — newer customers churn more
- `Contract type` is a dominant categorical driver — month-to-month customers churn significantly more
- `MonthlyCharges` weakly positively correlates with churn (0.19) — higher bills increase churn risk
- `TotalCharges` and `tenure` are highly collinear (0.83) — potential redundancy

### 3. Feature Engineering
Four new features were engineered:

| Feature | Description |
|---------|-------------|
| `AvgMonthlySpend` | `TotalCharges / (tenure + 1)` — historical average spend per month |
| `AddonCount` | Sum of 6 binary add-on service flags (0–6 scale) — measures service engagement |
| `TenureBucket` | Tenure segmented into New (0–12m), Mid (1–3yr), LongTerm (3yr+) |
| `HighSpender` | Binary flag — 1 if MonthlyCharges above median, 0 otherwise |

### 4. Preprocessing Pipeline (Scikit-learn ColumnTransformer)

```
ColumnTransformer
├── Numerical  → Median Imputation → StandardScaler
├── Categorical → Mode Imputation  → OneHotEncoder
└── Binary     → Mode Imputation   (already 0/1, no scaling needed)
```

### 5. SageMaker-Ready Output
- Preprocessed CSV saved with **no header** and **target column first** — required format for SageMaker built-in algorithms
- Upload to S3 using:

```python
import boto3
s3 = boto3.client('s3')
s3.upload_file(
    'telco_churn_preprocessed.csv',
    'your-bucket-name',
    'data/telco_churn_preprocessed.csv'
)
```

---

## Model — TabNet

[TabNet](https://arxiv.org/abs/1908.07442) is an attention-based deep learning architecture designed specifically for tabular data. It uses sequential attention to select relevant features at each decision step, providing both high performance and interpretability.

### Architecture Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_d` / `n_a` | 32 | Decision & attention embedding width |
| `n_steps` | 5 | Number of sequential attention steps |
| `gamma` | 1.3 | Feature reusage coefficient across steps |
| `lambda_sparse` | 1e-4 | Sparsity regularization on feature selection |
| `optimizer` | Adam | Learning rate: 2e-3, weight decay: 1e-5 |
| `scheduler` | StepLR | Step size: 10, gamma: 0.9 |
| `batch_size` | 256 | Training batch size |
| `virtual_batch_size` | 128 | Ghost batch normalization size |

### Class Imbalance Handling
- **Technique:** Sample Weights (no data modification)
- **Weight for Churn=1:** `neg / pos ≈ 2.77`
- **Weight for Churn=0:** `1.0`
- Each churn sample is treated as 2.77x more important during training

### Data Split
| Split | Size |
|-------|------|
| Train | 70% |
| Validation | 15% |
| Test | 15% |

All splits are **stratified** to preserve the class ratio.

---

## Results & Evaluation Metrics

### Model Performance

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.7804 |
| Average Precision | 0.5561 |

### ROC Curve
AUC of **0.78** means the model correctly ranks a random churner above a random non-churner 78% of the time. The curve bows clearly above the random classifier baseline.

### Precision-Recall Curve
AP of **0.55** on an imbalanced dataset (27% positive class) is meaningful — significantly above the random baseline of 0.27.

### Feature Importance (TabNet Attention)
Top predictive features identified by TabNet's attention mechanism:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | OnlineBackup_bin | 0.072 |
| 2 | Contract_Two year | 0.063 |
| 3 | Partner_No | 0.046 |
| 4 | TenureBucket_LongTerm | 0.046 |
| 5 | Dependents_Yes | 0.039 |
| 6 | PaymentMethod_Mailed check | 0.036 |
| 7 | MonthlyCharges | 0.033 |
| 8 | AddonCount | 0.030 |

> Note: `TenureBucket_LongTerm` and `AddonCount` are both engineered features — their presence in the top 10 validates the feature engineering decisions.

### SHAP Values
SHAP (SHapley Additive exPlanations) was used to provide class-specific feature importance:
- **Churn=1 SHAP:** Which features push predictions toward churning
- **Churn=0 SHAP:** Which features push predictions toward staying
- **Dot plot:** Shows direction and magnitude of each feature's impact per individual prediction

---

## How to Run

### Step 1 — Data Cleaning, EDA & Feature Engineering
```bash
jupyter nbconvert --to notebook --execute telco_churn_pipeline.py
# Or run cell by cell in Jupyter Notebook
```

### Step 2 — Model Training & Evaluation
```bash
jupyter nbconvert --to notebook --execute telco_churn_tabnet.py
# Or run cell by cell in Jupyter Notebook
```

### Step 3 — Upload to S3 (SageMaker)
```python
import boto3
s3 = boto3.client('s3')
s3.upload_file('telco_churn_preprocessed.csv', 'your-bucket-name', 'data/telco_churn_preprocessed.csv')
```

---

## Key Takeaways

- Contract type and service engagement are the dominant churn drivers
- Customers with shorter tenure and no add-on services are at the highest churn risk
- Gender has negligible predictive power — churn is driven by behavior, not demographics
- TabNet's attention mechanism naturally handles feature selection, complementing the engineered features
- With only 10 training epochs, ROC-AUC of 0.78 is a strong baseline — further tuning (more epochs, hyperparameter optimization) is expected to improve performance

---

## Future Work

- Increase training epochs and apply SageMaker Automatic Model Tuning (HPO) for hyperparameter optimization
- Implement cross-validation for more robust evaluation
- Explore SMOTE oversampling combined with class weights
- Deploy the trained model as a SageMaker endpoint for real-time inference
