# VBC Contract Simulator

An AI-powered Value-Based Care (VBC) contract simulation tool built with CMS public data. Simulates shared savings and penalty outcomes for any US hospital under configurable VBC contract terms, combining cost efficiency (MSPB) and quality (VBP TPS, HCAHPS, readmissions, star ratings) into a unified hospital value profile.

## Live Demo

> Deploy your own at [share.streamlit.io](https://share.streamlit.io) using this repo.

---

## Features

### 🔍 Hospital Lookup
Search any of ~5,400 US hospitals by name or state and view a full value profile card — MSPB ratio, VBP domain scores, star rating, readmission risks, HCAHPS patient experience highlights, and value quadrant placement.

### 📋 Contract Simulator
Define VBC contract parameters with interactive sliders (cost threshold, quality gates, savings share rate, penalty rate, contract type) and instantly project a bonus, penalty, or neutral outcome with estimated dollar impact and peer comparison.

### 🔮 What-If Analyzer
Ask "what if this hospital improved X by Y?" — slide MSPB, TPS, star rating, or readmission metrics and see how the contract outcome changes. Powered by SHAP waterfall charts showing the top improvement levers.

### 🗺️ Population Map
US choropleth map coloring hospitals by MSPB ratio, VBP TPS, or star rating. Filter by state, hospital type, ownership, and value archetype. Download filtered lists as CSV.

---

## Data Sources

All data is from [data.cms.gov](https://data.cms.gov) — public domain, no registration required.

| Dataset | CMS Page | Role in Model |
|---------|----------|---------------|
| Hospital General Information | [xubh-q36u](https://data.cms.gov/provider-data/dataset/xubh-q36u) | Master join table, star ratings |
| Medicare Spending Per Beneficiary (MSPB) | [rrqw-56er](https://data.cms.gov/provider-data/dataset/rrqw-56er) | Primary cost signal |
| Hospital VBP Total Performance Score | [ypbt-wvdk](https://data.cms.gov/provider-data/dataset/ypbt-wvdk) | Quality composite score |
| HCAHPS Patient Experience | [dgck-syfz](https://data.cms.gov/provider-data/dataset/dgck-syfz) | Patient experience dimension |
| Unplanned Hospital Visits / Readmissions | [632h-zaca](https://data.cms.gov/provider-data/dataset/632h-zaca) | Outcome dimension |

All five datasets are joined on **CCN (Facility ID / Provider ID)** into a master table of ~5,400 hospitals with 120+ features.

---

## ML Architecture

### XGBoost Payment Predictor
- **Target**: VBP payment adjustment % (derived from TPS percentile rank, scaled −2% to +2%)
- **Model**: XGBoost Regressor + Classifier (bonus / penalty / neutral direction)
- **Evaluation**: 5-fold CV RMSE: 0.037 | R²: 0.9998
- **Explainability**: SHAP values per hospital

### KMeans Value Clustering
- **Algorithm**: KMeans (k=6, best silhouette score = 0.514)
- **Features**: MSPB ratio, VBP TPS, star rating, composite readmission ratio
- **Archetypes**:

| Archetype | Cost | Quality | VBC Outlook |
|-----------|------|---------|-------------|
| High Value | ✅ Low | ✅ High | Bonus likely |
| Average Performers | ➡️ Mid | ➡️ Mid | Neutral |
| Cost Efficient / Lower Quality | ✅ Low | ➡️ Mid | Savings possible |
| High Quality / Higher Cost | ➡️ Mid | ✅ High | Neutral |
| Low Value | ❌ High | ❌ Low | Penalty risk |
| Emerging Performers | ❌ High | ❌ Low | Penalty risk |

### Contract Simulation Engine (rule-based)
Deterministic logic — no ML:
```
Quality gate passed  =  VBP TPS ≥ min_tps  AND  stars ≥ min_stars
Bonus                =  (1 − MSPB/target) × savings_share_rate   [if quality gate passed + MSPB < target]
Penalty              =  (MSPB − threshold) × penalty_rate         [if MSPB > penalty_threshold]
Neutral              =  neither condition met
```

---

## Project Structure

```
vbc-contract-simulator/
├── data/
│   ├── raw/                  ← CMS CSV downloads (gitignored)
│   └── processed/
│       ├── master_hospital.csv
│       └── features.pkl
├── models/
│   ├── xgb_payment_adj.pkl
│   └── kmeans_clusters.pkl
├── src/
│   ├── load_data.py          ← Phase 1: join 5 CMS datasets on CCN
│   ├── preprocess.py         ← Phase 2: feature engineering & normalization
│   ├── model.py              ← Phase 3: XGBoost + KMeans + SHAP
│   ├── simulator.py          ← Phase 4: deterministic contract engine
│   └── hcahps_labels.py      ← Human-readable HCAHPS measure labels
├── pages/
│   ├── 01_hospital_lookup.py
│   ├── 02_contract_simulator.py
│   ├── 03_what_if_analyzer.py
│   └── 04_population_map.py
├── app.py                    ← Streamlit entry point
├── requirements.txt
└── .streamlit/config.toml
```

---

## Local Setup

```bash
git clone https://github.com/Holin-Chen/vbc-contract-simulator.git
cd vbc-contract-simulator

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install --prefer-binary -r requirements.txt
streamlit run app.py
```

The processed data and trained models are included in the repo — no need to re-run the pipeline to use the app.

### Re-running the data pipeline (optional)

Download the 5 raw CSVs into `data/raw/` using the included script:
```powershell
.\download_data.ps1
```

Then run each phase:
```bash
python src/load_data.py      # Phase 1: join datasets → master_hospital.csv
python src/preprocess.py     # Phase 2: feature engineering → features.pkl
python src/model.py          # Phase 3: train ML models
python src/simulator.py      # Phase 4: run unit tests (5 tests)
```

---

## Stack

| Layer | Tools |
|-------|-------|
| Data | pandas, requests |
| ML | scikit-learn, XGBoost, imbalanced-learn |
| Explainability | SHAP |
| App | Streamlit (multi-page) |
| Charts & Maps | Plotly Express |
| Deployment | Streamlit Community Cloud |

---

## Disclaimer

For research and educational purposes only. Not financial, medical, or legal advice. Contract simulation outputs are approximations based on public CMS data and do not represent actual CMS payment determinations.
