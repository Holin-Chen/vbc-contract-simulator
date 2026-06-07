"""VBC Contract Simulator — Streamlit multi-page entry point."""

import streamlit as st

st.set_page_config(
    page_title="VBC Contract Simulator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.markdown("## 🏥 VBC Contract Simulator")
st.sidebar.markdown("AI-powered Value-Based Care contract simulation using CMS public data.")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ For research purposes only. Not financial or medical advice.")

st.title("VBC Contract Simulator")
st.markdown("""
Welcome to the **Value-Based Care Contract Simulator** — an AI-powered tool that simulates
shared savings and penalty outcomes for any US hospital using CMS public data.

### Navigation
Use the sidebar to navigate between pages:

| Page | Description |
|------|-------------|
| 🔍 Hospital Lookup | Search any hospital and view its value profile |
| 📋 Contract Simulator | Define contract parameters and project outcomes |
| 🔮 What-If Analyzer | See how metric improvements affect contract outcomes |
| 🗺️ Population Map | Explore hospital performance across the US |

### Data Sources
All data is from [data.cms.gov](https://data.cms.gov) — public domain, no registration required.
- Medicare Spending Per Beneficiary (MSPB)
- Hospital VBP Total Performance Score
- HCAHPS Patient Experience
- Hospital Readmissions Reduction Program
- Overall Hospital Quality Star Ratings

### Setup
Before using the app, run the data pipeline:
```bash
python src/load_data.py      # Phase 1: join datasets
python src/preprocess.py     # Phase 2: feature engineering
python src/model.py          # Phase 3: train ML models
python src/simulator.py      # Phase 4: verify simulation engine
```
""")
