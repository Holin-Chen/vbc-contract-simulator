# VBC Contract Simulator

## Project overview
AI-powered Value-Based Care contract simulation tool using CMS public data. Simulates shared savings/penalty outcomes for US hospitals under configurable VBC contract terms. Combines cost (MSPB) and quality (VBP TPS, HCAHPS, readmissions, star ratings) into a unified hospital value profile.

## Data files (data/raw/ — all from data.cms.gov, public domain)
- Hospital_General_Information.csv ← master join table (CCN = join key)
- Medicare_Spending_Per_Beneficiary_Hospital.csv
- VBP_Hospital_TPS.csv
- HCAHPS_Hospital.csv
- Unplanned_Hospital_Visits_Provider_Data.csv

Processed: data/processed/master_hospital.csv (after Phase 1)

## Key column reference
- MSPB_Ratio: episode cost vs national median (1.0 = average)
- Total_Performance_Score: VBP quality score 0-100
- Hospital_overall_rating: 1-5 stars
- Excess_Readmission_Ratio: per condition (1.0 = expected; >1.0 = worse than expected)
- Payment_Adjustment_Percent: VBP bonus (positive) or penalty (negative)

## Architecture
- src/load_data.py — join 5 CMS datasets on CCN
- src/preprocess.py — feature engineering, normalization
- src/model.py — XGBoost predictor + KMeans clusters + SHAP
- src/simulator.py — deterministic contract simulation engine
- pages/ — 4 Streamlit pages (lookup, simulator, what-if, map)
- app.py — Streamlit multi-page entry point

## Stack
Python 3.11, pandas, numpy, scikit-learn, xgboost, shap, streamlit, plotly, joblib, imbalanced-learn

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Coding conventions
- Always run scripts after writing and print confirmation of output shape/metrics
- All file paths must be relative to project root
- Use @st.cache_data for all Streamlit data loading functions
- Wrap all model predictions in try/except with user-friendly error messages
- Disclaimer required: 'For research purposes only. Not financial advice.'
- Unit tests required for simulator.py (5 test cases minimum)

## Contract simulation logic
- Quality gate passed = hospital_tps >= min_quality_score AND star_rating >= min_stars
- Bonus = (1 - mspb_ratio) * savings_share_rate  [if quality gate passed + mspb < target]
- Penalty = (mspb_ratio - 1) * penalty_rate  [if mspb > penalty_threshold]
- Neutral = neither condition met
