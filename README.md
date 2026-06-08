# E-Commerce Customer Segmentation

An end-to-end customer segmentation project on the **UCI Online Retail II** dataset (~1M transactions, 5,862 customers).
The project moves from raw transactional data to a deployable ML classifier that assigns new customers to business segments in real time.

---

## Business Problem

A UK-based online retailer has thousands of customers ranging from one-time low-spend buyers to high-value wholesale accounts spending £500K+.
Treating all customers the same with a single marketing strategy is inefficient.

**Goal:** Segment customers into meaningful groups using RFM analysis and clustering, then build a classifier to automate segment assignment for new customers.

---

## Dataset

**UCI Online Retail II** — Transactional data from a UK-based online retailer (2009–2011)

| Property | Value |
|----------|-------|
| Raw rows | 1,067,371 |
| Clean rows | 776,941 |
| Unique customers | 5,862 |
| Unique products | 4,629 |
| Date range | Dec 2009 — Dec 2011 |
| Source | [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) |

---

## Project Structure

    ecommerce-customer-segmentation/
    │
    ├── data/
    │   ├── raw/                        # Original dataset (not tracked)
    │   └── processed/                  # Cleaned & engineered data
    │       ├── retail_cleaned.csv
    │       ├── rfm.csv
    │       ├── rfm_scaled.csv
    │       ├── rfm_scored.csv
    │       ├── rfm_clustered.csv
    │       └── rfm_final.csv
    │
    ├── notebooks/
    │   ├── 01_eda_cleaning.ipynb
    │   ├── 02_rfm_feature_engineering.ipynb
    │   ├── 03_classical_clustering.ipynb
    │   ├── 04_segment_profiling.ipynb
    │   └── 05_predictive_layer.ipynb
    │
    ├── models/
    │   └── segment_classifier.pkl
    │
    ├── reports/figures/
    ├── requirements.txt
    └── README.md

---

## Pipeline

| # | Notebook | Description | Key Output |
|---|----------|-------------|------------|
| 1 | EDA & Cleaning | Missing values, duplicates, returns, seasonality, retention analysis | 776,941 clean transactions |
| 2 | RFM Feature Engineering | Recency, Frequency, Monetary computation, log scaling, quantile scoring | 5,862 customer RFM profiles |
| 3 | Classical Clustering | K-Means, Hierarchical, DBSCAN with evaluation metrics | K-Means K=4 selected |
| 4 | Segment Profiling | Cluster profiling, radar charts, business recommendations | 4 named segments |
| 5 | Predictive Layer | Random Forest classifier for real-time segment assignment | 99.23% accuracy |

---

## Key EDA Findings

- **Seasonality:** Revenue peaks every November (Christmas), drops sharply in December (incomplete month)
- **Geography:** UK dominates (~85% of revenue ~£14.5M), top international markets are EIRE, Netherlands, Germany, France
- **Products:** Top 21% of products drive 80% of revenue — clear Pareto effect
- **Behaviour:** Peak activity Thu 10am–1pm, zero Saturday — strong B2B wholesale characteristics
- **Retention:** 63% year-on-year customer retention rate

---

## Customer Segments

| Segment | Customers | % | Avg Recency | Avg Frequency | Avg Monetary | Strategy |
|---------|-----------|---|-------------|---------------|--------------|----------|
| 🏆 Champions | 1,189 | 20.3% | 28 days | 19 orders | £10,633 | Reward, upsell, protect |
| 🌱 New & Promising | 1,249 | 21.3% | 28 days | 3 orders | £840 | Nurture, incentivise repeat orders |
| ⚠️ At Risk | 1,458 | 24.9% | 228 days | 5 orders | £1,913 | Win-back campaigns |
| ❌ Lost | 1,966 | 33.5% | 395 days | 1.4 orders | £314 | Minimal spend, analyse churn |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Algorithm | Random Forest (100 trees) |
| Accuracy | 99.23% |
| CV Accuracy (5-fold) | 98.29% ± 0.51% |
| Most important feature | Recency (0.378) |

---

## Tech Stack

`Python` `Pandas` `NumPy` `Scikit-learn` `Matplotlib` `Seaborn` `Plotly` `Joblib`

---

## Setup

```bash
git clone https://github.com/CaptainLevy/ecommerce-customer-segmentation.git
cd ecommerce-customer-segmentation
pip install -r requirements.txt
```

Place `online_retail_II.csv` in `data/raw/` then run notebooks in order.

---

## Author

**Ankit Kumar** — M.Sc. Statistics  
[GitHub](https://github.com/CaptainLevy)