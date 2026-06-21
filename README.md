# IPO Underpricing Prediction — ML Capstone

**CS 0451 Machine Learning · Middlebury College · Spring 2026**

A machine learning pipeline that predicts whether an IPO will be **underpriced, fairly priced, or overpriced** before opening day — combining financial data, macroeconomic indicators, and NLP-extracted signals from SEC S-1 filings.

---

## The Problem

IPO underpricing costs companies billions annually. When Airbnb went public, its stock opened at nearly double its offer price — leaving over $3 billion on the table. On average, IPOs are underpriced by 10–20%. This project asks: **can ML detect the patterns behind mispricing before trading begins?**

---

## Data Sources

| Source | Features |
|---|---|
| Bloomberg Terminal | Offer price, deal size, shares outstanding, industry, lead underwriter |
| FRED API | Interest rates, CPI, unemployment — matched to each IPO date |
| SEC EDGAR S-1 Filings | NLP-extracted risk scores, growth signals, business complexity via GPT-4o-mini |

Integrating structured financial data, macroeconomic time series, and unstructured regulatory text into a single feature pipeline was the core engineering challenge of this project.

---

## Results

### Binary Classification (Full Dataset, n=6,110)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 58% | 0.58 |
| Neural Network | 69% | 0.64 |
| Random Forest | 70% | 0.65 |
| **XGBoost** | **71%** | **0.70** |

XGBoost beat the naive baseline (always predicting "underpriced" = 60% accuracy) by 11 points and was the most balanced across classes. A naive baseline that always predicts "underpriced" would achieve 60% accuracy without learning anything useful — Macro F1 tells the real story.

### 3-Class Classification (Full Dataset)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 50% | 0.45 |
| Neural Network | 53% | 0.51 |
| Random Forest | 63% | 0.49 |
| XGBoost | 60% | 0.55 |

### Top Features (XGBoost — Binary)

Offer Price dominated with an importance score of ~0.18, nearly double the next feature. Market Cap at Offer and GDP followed at ~0.09 each, with CPI and bulge-bracket underwriter status rounding out the top 5 — confirming that pricing decisions and the macro environment matter more than firm-specific characteristics.

### Risk Feature Impact (XGBoost on SEC S-1 Subset, n=862)

Adding GPT-4o-mini risk scores from S-1 filings produced small, consistent gains across precision, recall, and F1 — but no single improvement exceeded 3 percentage points. A larger dataset may reveal stronger effects.

> 📓 Full confusion matrices, feature importance plots, and per-class breakdowns in [`report.ipynb`](report.ipynb)

---

## Key Technical Highlights

- **End-to-end feature pipeline** combining structured financial data, macroeconomic time series, and unstructured text
- **NLP feature extraction** from SEC S-1 filings using GPT-4o-mini to score regulatory risk, competitive risk, financial risk, and overall risk on a 1–10 scale
- **Four model types** — Logistic Regression (custom PyTorch), Neural Network (custom PyTorch MLP), Random Forest (custom from scratch), and XGBoost — all trained and evaluated on identical 80/20 stratified splits
- **Class imbalance handling** via class weights across all models (60% of IPOs underpriced in full dataset)
- **Feature importance analysis** identifying offer price, market cap, and macro conditions as the strongest predictors

---

## Project Structure

```
ipologists/
├── data/               # Raw and processed datasets
├── notebooks/          # Per-model Jupyter notebooks
├── scripts/            # Data collection (Bloomberg, FRED, SEC EDGAR)
├── src/                # Core pipeline modules and saved models
├── report.ipynb        # Full analysis, results, and visualizations
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

**Mac only (required for XGBoost parallel processing):**
```bash
brew install libomp
```

**Create a `.env` file with your API keys:**
```
FRED_API_KEY=your_key
OPENAI_API_KEY=your_key
```

---

## Quick Demo

```python
import pickle, pandas as pd
from src.pipeline import features, prepare_xgboost

full = pd.read_csv("data/final/dataset_full.csv")
with open("src/models/saved_models/xgBoost/xgb_binary_full.pkl", "rb") as f:
    model = pickle.load(f)

_, X_test, _, y_test = prepare_xgboost(full, "underpriced", features)
preds = model.predict(X_test)  # 71% accuracy on held-out test set
```

---

## Authors

[William Zambito](https://github.com/ZambitoW) · Anna Kester  
Middlebury College — Computer Science & Economics

---

*For educational and research purposes only. Not financial advice.*
