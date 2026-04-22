# 🛍️ ShopSense AI — Product Recommendation System

> **Capstone Project · Week 12**  
> A production-grade Factorization Machine recommendation engine built on the UCI Online Retail dataset, with Generative AI visual enhancements.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Dataset Description](#3-dataset-description)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Factorization Machine Model](#5-factorization-machine-model)
6. [Recommendation Engine](#6-recommendation-engine)
7. [Generative AI Enhancement](#7-generative-ai-enhancement-capstone)
8. [Streamlit Dashboard](#8-streamlit-dashboard)
9. [Installation & Quick Start](#9-installation--quick-start)
10. [Evaluation Results](#10-evaluation-results)
11. [AWS Bedrock Integration](#11-aws-bedrock-integration)
12. [Project File Structure](#12-project-file-structure)
13. [References](#13-references)

---

## 1. Project Overview

This project implements an end-to-end **product recommendation system** inspired by real-world e-commerce platforms (Amazon, Alibaba). It uses **Factorization Machines** — the same algorithm family used by Alibaba's recommendation engine — to learn latent user and product embeddings from historical purchase data, then leverages **Generative AI** to visually present recommendations with AI-generated product images.

### Key Capabilities

| Feature | Description |
|---|---|
| **Personalised Recommendations** | Top-N product recommendations per customer |
| **Item Similarity** | Find products most similar to any given item |
| **Implicit Feedback** | No explicit ratings needed — uses purchase quantity as signal |
| **Side Features** | Product descriptions (TF-IDF) and user geography (one-hot) enrich the FM |
| **AI Product Images** | HuggingFace SDXL generates visual product cards |
| **Interactive Dashboard** | Streamlit dark-mode UI for real-time exploration |
| **AWS Bedrock Ready** | CLI-based Titan Image Generator integration for cloud deployment |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│                                                                 │
│   data.csv ──► preprocess.py ──► data_clean.csv                │
│                                  encoders.pkl                   │
│                                  user_item_matrix.npz           │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     MODEL TRAINING                               │
│                                                                 │
│   data_clean.csv ──► train_fm.py ──► fm_model.pkl              │
│   (+ TF-IDF item features)          (LightFM WARP)             │
│   (+ Country user  features)                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  INFERENCE & SERVING                             │
│                                                                 │
│   fm_model.pkl ──► recommend.py ──► recommendations_output.csv │
│                  ──► app.py      ──► Streamlit Dashboard        │
│                  ──► ai_visual.py──► AI Product Images          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Dataset Description

**Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)  
**Provider:** Dr Daqing Chen, London South Bank University

| Property | Value |
|---|---|
| File | `data.csv` |
| Encoding | ISO-8859-1 |
| Raw Rows | 541,909 |
| Columns | 8 |
| Time Range | Dec 2010 – Dec 2011 |
| Market | UK-based online gift retailer |

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `InvoiceNo` | string | Unique invoice ID; prefix `C` = cancellation |
| `StockCode` | string | Unique product identifier |
| `Description` | string | Product name / description |
| `Quantity` | int | Units purchased (negative = return) |
| `InvoiceDate` | datetime | Transaction timestamp |
| `UnitPrice` | float | Price per unit in GBP (£) |
| `CustomerID` | string | Unique customer identifier |
| `Country` | string | Customer's country |

### Raw Data Quality Issues

| Issue | Count | Action |
|---|---|---|
| Missing `CustomerID` | 135,080 rows | **Dropped** — can't personalise without identity |
| Cancellation invoices | 9,288 rows | **Removed** — negative signal, not a purchase |
| Negative quantities | 10,624 rows | **Removed** — returns/adjustments |
| Zero/negative prices | 2,517 rows | **Removed** — free samples / data errors |
| Missing descriptions | 1,454 rows | **Filled** with `"UNKNOWN"` |

---

## 4. Preprocessing Pipeline

All steps are implemented in **`preprocess.py`** and run in sequence. Each step is logged to the console with row counts so you can audit what was removed and why.

### Step-by-Step Breakdown

#### Step 1 — Load Raw Data
```python
df = pd.read_csv("data.csv", encoding="ISO-8859-1",
                 dtype={"CustomerID": str, "InvoiceNo": str})
```
- **ISO-8859-1 encoding** is used (not UTF-8) because the dataset contains special Latin characters common in UK product names.
- `CustomerID` and `InvoiceNo` are loaded as strings to preserve leading zeros and the `C` prefix.

#### Step 2 — Drop Missing CustomerID
```python
df = df.dropna(subset=["CustomerID"])
```
- **Rationale:** 135,080 rows (~25%) have no CustomerID, representing anonymous/guest sessions.
- Personalized recommendations require a stable user identity. These rows cannot be used.
- **Impact:** 541,909 → 406,829 rows.

#### Step 3 — Remove Cancellations
```python
df = df[~df["InvoiceNo"].str.startswith("C", na=False)]
```
- **Rationale:** Invoices prefixed with `C` are credit notes (refunds/cancellations).
- Including them would create *negative* purchase signals, misleading the model.
- **Impact:** ~9,288 rows removed.

#### Step 4 — Remove Invalid Quantities
```python
df = df[df["Quantity"] > 0]
```
- **Rationale:** Negative quantities represent items returned from previous orders.
- Zero quantities indicate adjustments. Neither represents a real purchase event.
- **Impact:** ~10,624 rows removed.

#### Step 5 — Remove Invalid Prices
```python
df = df[df["UnitPrice"] > 0]
```
- **Rationale:** Items with `UnitPrice == 0` are free samples or manual adjustments.
- These would distort the `TotalPrice` calculation used in the interaction signal.
- **Impact:** ~2,517 rows removed.

#### Step 6 — Clean Text Fields
```python
df["Description"] = df["Description"].fillna("UNKNOWN").str.strip().str.upper()
df["StockCode"]   = df["StockCode"].str.strip().str.upper()
df["Country"]     = df["Country"].str.strip()
```
- **Rationale:** Inconsistent casing and whitespace would cause the same item to be treated as multiple distinct items by the label encoder and TF-IDF vectorizer.

#### Step 7 — Parse Dates & Temporal Features
```python
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], infer_datetime_format=True)
df["Year"]       = df["InvoiceDate"].dt.year
df["Month"]      = df["InvoiceDate"].dt.month
df["DayOfWeek"]  = df["InvoiceDate"].dt.dayofweek
```
- **Rationale:** Temporal features (Year, Month, DayOfWeek) enable downstream time-series analysis and could be used as FM context features in a more advanced version.

#### Step 8 — Derive TotalPrice
```python
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
```
- **Uses:** Customer spend analysis, EDA charts in the dashboard.

#### Step 9 — Aggregate to User-Item Pairs
```python
agg = df.groupby(["CustomerID", "StockCode", "Description", "Country"]).agg(
    TotalQuantity = ("Quantity",   "sum"),
    TotalSpend    = ("TotalPrice", "sum"),
    PurchaseCount = ("InvoiceNo",  "nunique"),
).reset_index()

agg["ImplicitRating"] = np.log1p(agg["TotalQuantity"])
```
- **Why aggregate?** The FM model works on user-item interaction pairs, not individual transactions. A customer buying the same product 5 times across 3 invoices should count once, with a stronger signal.
- **Why `log1p`?** Raw quantities are heavily right-skewed (some bulk buyers purchase 1000+ units). `log1p` compresses the scale while preserving the ordering — customers who buy more still score higher, but extreme values don't dominate the matrix.

#### Step 10 — Label Encoding
```python
user_enc = LabelEncoder()
item_enc = LabelEncoder()
df["UserIdx"] = user_enc.fit_transform(df["CustomerID"])
df["ItemIdx"] = item_enc.fit_transform(df["StockCode"])
```
- **Why needed?** LightFM's sparse matrix requires contiguous integer indices for rows (users) and columns (items).
- Encoders are saved to `encoders.pkl` so they can be reused during inference without re-fitting.

#### Step 11 — Build Sparse Matrix
```python
mat = csr_matrix(
    (df["ImplicitRating"].values, (df["UserIdx"].values, df["ItemIdx"].values)),
    shape=(n_users, n_items),
)
```
- Format: **Compressed Sparse Row (CSR)** — optimal for row-slicing operations during model training.
- The matrix has shape `(n_users × n_items)` with values = `ImplicitRating`.
- Density is typically **~0.3%** — highly sparse, which is exactly where FM models excel.

#### Step 12 — Save Artefacts
Produces three output files:
- `data_clean.csv` — cleaned interaction pairs
- `encoders.pkl` — user and item LabelEncoders
- `user_item_matrix.npz` — sparse interaction matrix

---

## 5. Factorization Machine Model

### What is a Factorization Machine?

A **Factorization Machine (FM)** generalises matrix factorisation by learning pairwise feature interactions through latent vector dot products. Given a feature vector **x**, the FM prediction is:

```
ŷ(x) = w₀ + Σᵢ wᵢxᵢ + Σᵢ Σⱼ>ᵢ <vᵢ, vⱼ> · xᵢxⱼ
```

Where:
- `w₀` = global bias
- `wᵢ` = feature bias for feature `i`
- `vᵢ` = latent vector of dimension `k` for feature `i`
- `<vᵢ, vⱼ>` = dot product of latent vectors (captures interaction between features `i` and `j`)

The key insight: instead of learning a separate weight for every possible feature pair (which would be `O(n²)` parameters), FMs factor interactions into `O(nk)` latent vectors. This enables generalisation to unseen feature combinations.

### Why LightFM?

We use **LightFM** — a hybrid FM that seamlessly combines collaborative filtering with content-based features:

| Property | Detail |
|---|---|
| Library | [lyst/lightfm](https://github.com/lyst/lightfm) |
| Loss Function | **WARP** (Weighted Approximate-Rank Pairwise) |
| Embedding Dim | 64 components |
| Item Features | Identity one-hot + TF-IDF on Description (100 terms) |
| User Features | Identity one-hot + Country one-hot |
| Regularisation | L2 item_alpha=1e-6, user_alpha=1e-6 |
| Training | 30 epochs, 4 threads |

### WARP Loss

WARP optimises directly for **Precision@K** — exactly what matters for the top-of-list recommendations shown to users. For each positive interaction, WARP:

1. Samples a negative item (one the user hasn't bought)
2. If the negative item is ranked *above* the positive item, it is used as a training example
3. Scales the gradient by an approximation of the rank violation severity

This makes WARP far superior to BPR or point-wise losses for implicit-feedback scenarios.

### Item Features

```
Item Feature Matrix (n_items × [n_items + 100])
 ┌──────────────────────┬────────────────────────────┐
 │  Identity Block      │  TF-IDF (Description)      │
 │  (item bias)         │  (top 100 bigrams/unigrams) │
 │  n_items × n_items   │  n_items × 100              │
 └──────────────────────┴────────────────────────────┘
```

The identity block allows the model to learn per-item biases (some items are globally popular). The TF-IDF block enables the model to generalise to unseen items that share descriptive terms.

### User Features

```
User Feature Matrix (n_users × [n_users + n_countries])
 ┌──────────────────────┬────────────────────────────┐
 │  Identity Block      │  Country One-Hot            │
 │  (user bias)         │  (captures geo patterns)    │
 └──────────────────────┴────────────────────────────┘
```

---

## 6. Recommendation Engine

`recommend.py` exposes three main functions:

### `get_top_n_recommendations(customer_id, n=10)`
1. Looks up the customer's integer index via the saved encoder
2. Calls `model.predict(user_idx, all_item_indices, ...)` — scores all N items in one pass
3. Masks out already-purchased items (`score = -∞`)
4. Returns the top-N items sorted by score, with normalised relevance percentage

### `get_similar_items(stock_code, n=10)`
1. Retrieves the item's latent embedding from `model.get_item_representations()`
2. Computes **cosine similarity** between the query item and all other items
3. Returns the top-N most similar products

> **Why cosine similarity?** The FM's latent vectors encode semantic relationships between items. Items co-purchased together or with similar descriptions will have embeddings pointing in similar directions in the latent space.

### `batch_export(n=10)`
Scores all customers and writes `recommendations_output.csv` — useful for:
- Offline email campaign targeting
- A/B test pre-computation
- Analytics and reporting

---

## 7. Generative AI Enhancement (Capstone)

`ai_visual.py` adds a visual layer using **Generative AI image synthesis**.

### Architecture

```
Product Description Text
         │
         ▼
   Prompt Engineering
   "A high-quality professional product photo of {desc},
    white background, studio lighting, e-commerce style, 4K"
         │
    ┌────┴─────────────────────────────────────────┐
    │                                               │
    ▼ (if HF_TOKEN set)                            ▼ (fallback)
HuggingFace Inference API                   Styled Placeholder
stabilityai/sdxl-base-1.0                   (gradient + text)
    │
    ▼
Disk Cache (image_cache/)
    │
    ▼
PIL Image → base64 → Streamlit card
```

### HuggingFace Setup

```bash
# 1. Get a free token at https://huggingface.co/settings/tokens
# 2. Set environment variable:

# PowerShell
$env:HF_TOKEN = "hf_your_token_here"

# Then run:
streamlit run app.py
```

### AWS Bedrock Titan (CLI-based)

When AWS credentials are available, use the CLI to invoke Bedrock's Titan Image Generator:

```python
# Generate the CLI command for any product:
from ai_visual import bedrock_cli_command
cmd = bedrock_cli_command("WHITE HANGING HEART T-LIGHT HOLDER")
print(cmd)
```

This outputs an `aws bedrock-runtime invoke-model` command you can run in your terminal:

```bash
aws bedrock-runtime invoke-model \
  --model-id amazon.titan-image-generator-v1 \
  --body '{"taskType":"TEXT_IMAGE","textToImageParams":{"text":"..."},...}' \
  --cli-binary-format raw-in-base64-out \
  bedrock_output.json
```

Then decode the response:

```python
from ai_visual import generate_via_bedrock_response
img = generate_via_bedrock_response("bedrock_output.json")
img.save("product.png")
```

> **Note:** Bedrock is available in regions `us-east-1`, `us-west-2`. Requires IAM permissions: `bedrock:InvokeModel` on `arn:aws:bedrock:*::foundation-model/amazon.titan-image-generator-v1`.

### Image Caching

All generated images are cached to `image_cache/` using an MD5 hash of the description as the filename key. This avoids redundant API calls on repeated dashboard views.

---

## 8. Streamlit Dashboard

`app.py` provides a fully interactive dark-mode UI built with Streamlit, Plotly, and custom CSS.

### Features

| Panel | Description |
|---|---|
| **Customer Selector** | Dropdown over all 4,372 customers |
| **Recommendation Cards** | Grid of N product cards with AI images, rank badge, score bar |
| **Purchase History** | Table of what the customer has previously bought |
| **Similar Items Tab** | Embedding-based item similarity explorer |
| **Data Insights Tab** | EDA charts (top products, revenue by country, monthly trend, frequency distribution) |
| **Model Metrics Sidebar** | Live AUC, Precision@K, Recall@K, NDCG@K display |
| **AI Image Toggle** | Switch between AI-generated images and icon placeholders |

### Running the Dashboard

```bash
streamlit run app.py
```

---

## 9. Installation & Quick Start

### Prerequisites

- Python 3.10+
- Git (for LightFM installation from source)
- Visual C++ Build Tools on Windows (optional, for compilation)

### Install Dependencies

```bash
# LightFM must be installed from GitHub on Python 3.10+ (Windows)
pip install git+https://github.com/lyst/lightfm.git

# Remaining dependencies
pip install streamlit huggingface_hub pillow plotly seaborn \
            scipy scikit-learn numpy pandas matplotlib requests
```

### Run the Full Pipeline

```bash
# Step 1 — Clean and preprocess the data (~30–60 seconds)
python preprocess.py

# Step 2 — Train the FM model (~3–5 minutes)
python train_fm.py

# Step 3 — Generate batch recommendations
python recommend.py

# Step 4 (optional) — Test AI image generation
$env:HF_TOKEN = "hf_your_token_here"   # PowerShell
python ai_visual.py

# Step 5 — Launch the dashboard
streamlit run app.py
```

### Expected Runtimes

| Step | Estimated Time |
|---|---|
| `preprocess.py` | 45–90 seconds |
| `train_fm.py` | 3–6 minutes |
| `recommend.py` (batch) | 2–4 minutes |
| `app.py` startup | <5 seconds |

---

## 10. Evaluation Results

The model is evaluated on a 20% held-out test set using four standard recommendation metrics:

| Metric | Description | Typical Value |
|---|---|---|
| **AUC** | Area Under the ROC curve. Probability that a positive item is ranked above a random negative. | > 0.85 |
| **Precision@10** | Fraction of the top-10 recommended items that are actually relevant. | > 0.08 |
| **Recall@10** | Fraction of relevant items that appear in the top-10. | > 0.04 |
| **NDCG@10** | Normalised Discounted Cumulative Gain. Rewards placing relevant items higher in the list. | > 0.10 |

> **Note:** Implicit feedback datasets typically show lower Precision/Recall values than explicit rating datasets. An AUC > 0.85 and NDCG > 0.10 indicate a well-performing model for e-commerce implicit data.

### Interpreting AUC

- AUC = 0.5 → random baseline (no learning)
- AUC = 0.7 → reasonable
- AUC = 0.85+ → strong performance (expected for this dataset)

---

## 11. AWS Bedrock Integration

The system is designed to be cloud-deployable on AWS using Bedrock for image generation. Since direct SDK calls require credentials accessible at runtime, we provide a **CLI-based pattern** that works with any configured AWS profile.

### Required IAM Permissions

```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel"],
  "Resource": "arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-image-generator-v1"
}
```

### CLI Invocation Pattern

```bash
# Generate image for product
aws bedrock-runtime invoke-model \
  --model-id amazon.titan-image-generator-v1 \
  --region us-east-1 \
  --body '{
    "taskType": "TEXT_IMAGE",
    "textToImageParams": {
      "text": "A professional product photo of white hanging heart t-light holder, white background, studio lighting"
    },
    "imageGenerationConfig": {
      "numberOfImages": 1,
      "height": 512,
      "width": 512,
      "cfgScale": 8.0
    }
  }' \
  --cli-binary-format raw-in-base64-out \
  bedrock_output.json

# Decode via Python
python -c "
import json, base64
from PIL import Image
import io
data = json.load(open('bedrock_output.json'))
img  = Image.open(io.BytesIO(base64.b64decode(data['images'][0])))
img.save('product.png')
print('Saved product.png')
"
```

---

## 12. Project File Structure

```
Week_12/
│
├── data.csv                      # Raw UCI Online Retail dataset (unchanged)
│
├── preprocess.py                 # Step-by-step data cleaning pipeline
├── train_fm.py                   # LightFM WARP model training & evaluation
├── recommend.py                  # Inference engine (top-N, similarity, batch)
├── ai_visual.py                  # Generative AI image module (HF + Bedrock)
├── app.py                        # Streamlit interactive dashboard
│
├── data_clean.csv                # [Generated] Cleaned user-item interactions
├── encoders.pkl                  # [Generated] LabelEncoders for users & items
├── user_item_matrix.npz          # [Generated] Sparse interaction matrix
├── fm_model.pkl                  # [Generated] Trained LightFM model bundle
├── recommendations_output.csv    # [Generated] Batch recommendations CSV
│
├── image_cache/                  # [Generated] Cached AI product images
│   └── *.png
│
└── README.md                     # This file
```

---

## 13. References

| Source | Description |
|---|---|
| [LightFM Paper (Kula, 2015)](https://arxiv.org/abs/1507.08439) | Metadata Embeddings for User and Item Cold-start Recommendations |
| [Factorization Machines (Rendle, 2010)](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) | Original FM paper |
| [WARP Loss (Weston et al., 2011)](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf) | Optimising for ranking via WARP |
| [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) | Source dataset |
| [HuggingFace SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | Image generation model |
| [AWS Bedrock Titan](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-image-models.html) | Cloud image generation |

---

*Built as a capstone project for the Cloudelligent AI/ML Internship — Week 12.*
