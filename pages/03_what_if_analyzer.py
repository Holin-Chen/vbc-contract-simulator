"""Page 3: What-If Analyzer with SHAP explainability."""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.simulator import what_if_simulate, simulate_contract
from src.hcahps_labels import feature_label

st.set_page_config(page_title="What-If Analyzer", page_icon="🔮", layout="wide")
st.sidebar.caption("⚠️ For research purposes only. Not financial or medical advice.")


@st.cache_data
def load_master():
    df = pd.read_csv("data/processed/master_hospital.csv", dtype=str)
    df["CCN"] = df["CCN"].str.zfill(6)
    for c in ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating",
              "Payment_Adjustment_Percent", "Composite_Readmission"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@st.cache_resource
def load_model():
    model_path = Path("models/xgb_payment_adj.pkl")
    feat_path = Path("data/processed/features.pkl")
    if not model_path.exists() or not feat_path.exists():
        return None, None, None
    data = joblib.load(model_path)
    feat_data = joblib.load(feat_path)
    return data["regressor"], data["feature_cols"], feat_data["X"]


st.title("🔮 What-If Analyzer")
st.markdown("Simulate how metric improvements would change this hospital's contract outcome.")

try:
    df = load_master()
except FileNotFoundError:
    st.error("master_hospital.csv not found. Run `python src/load_data.py` first.")
    st.stop()

name_col = next((c for c in ["Facility Name", "Hospital Name"] if c in df.columns), None)
state_col = "State" if "State" in df.columns else None

# ── Hospital selection ────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Hospital Selection")
    if state_col:
        sel_state = st.selectbox("State", ["All"] + sorted(df[state_col].dropna().unique().tolist()))
        filtered = df if sel_state == "All" else df[df[state_col] == sel_state]
    else:
        filtered = df
    search = st.text_input("Search hospital name")
    if search and name_col:
        filtered = filtered[filtered[name_col].str.contains(search, case=False, na=False)]
    if name_col and not filtered.empty:
        selected_name = st.selectbox("Hospital", filtered[name_col].dropna().tolist())
        row = filtered[filtered[name_col] == selected_name].iloc[0]
        ccn = row["CCN"]
    else:
        st.stop()

    st.markdown("---")
    st.subheader("Contract Parameters")
    target_mspb = st.slider("Cost target (MSPB)", 0.80, 1.10, 1.00, 0.01)
    min_tps = st.slider("Min VBP TPS", 0, 100, 45)
    min_stars = st.slider("Min Stars", 1, 5, 3)
    savings_rate = st.slider("Savings share rate", 0.10, 0.80, 0.50, 0.05)
    penalty_threshold = st.slider("Penalty threshold", 1.00, 1.20, 1.05, 0.01)
    penalty_rate_val = st.slider("Penalty rate", 0.01, 0.10, 0.02, 0.005)
    benchmark_vol = st.number_input("Est. annual Medicare spend ($)", value=50_000_000, step=1_000_000)

contract_params = {
    "min_quality_score": float(min_tps),
    "min_stars": int(min_stars),
    "target_mspb": float(target_mspb),
    "penalty_threshold": float(penalty_threshold),
    "savings_share_rate": float(savings_rate),
    "penalty_rate": float(penalty_rate_val),
    "benchmark_volume": float(benchmark_vol),
}

# ── Improvement sliders ───────────────────────────────────────────────────────
st.subheader(f"Hypothetical Improvements: {selected_name}")
c1, c2, c3, c4 = st.columns(4)
with c1:
    mspb_delta = st.slider("MSPB Ratio change", -0.30, 0.30, 0.0, 0.01,
                           help="Negative = cost improvement (e.g. -0.05 = 5% more efficient)")
with c2:
    tps_delta = st.slider("VBP TPS change", -20, 20, 0, 1)
with c3:
    stars_delta = st.slider("Star Rating change", -2, 2, 0, 1)
with c4:
    readmit_delta = st.slider("Readmission ratio change", -0.20, 0.20, 0.0, 0.01)

improvements = {}
if mspb_delta != 0:
    improvements["mspb"] = float(mspb_delta)
if tps_delta != 0:
    improvements["tps"] = float(tps_delta)
if stars_delta != 0:
    improvements["stars"] = float(stars_delta)

result = what_if_simulate(ccn, improvements, contract_params, df)
baseline = result["baseline"]
improved = result["improved"]

# ── Side-by-side comparison ───────────────────────────────────────────────────
st.markdown("---")
bc, ic = st.columns(2)

def outcome_box(res, label):
    outcome = res.get("outcome", "unknown")
    adj = res.get("adjustment_pct", 0)
    est = res.get("savings_pool_est", 0)
    if outcome == "bonus":
        st.success(f"**{label}**: ✅ BONUS +{adj:.3f}% | Est. ${est:,.0f}")
    elif outcome == "penalty":
        st.error(f"**{label}**: ❌ PENALTY {adj:.3f}% | Est. -${abs(est):,.0f}")
    else:
        st.info(f"**{label}**: ➡️ NEUTRAL")

with bc:
    st.subheader("Current State")
    outcome_box(baseline, "Baseline")
    st.metric("MSPB Ratio", f"{baseline.get('mspb', np.nan):.3f}" if not pd.isna(baseline.get("mspb", np.nan)) else "N/A")
    st.metric("VBP TPS", f"{baseline.get('tps', np.nan):.1f}" if not pd.isna(baseline.get("tps", np.nan)) else "N/A")
    st.metric("Stars", str(int(baseline.get("stars", 0))) if not pd.isna(baseline.get("stars", np.nan)) else "N/A")
    qg = baseline.get("quality_gate_results", {})
    for gname, gate in qg.items():
        icon = "✅" if gate["passed"] else "❌"
        st.markdown(f"{icon} {gname}: {gate.get('value', 'N/A')}")

with ic:
    st.subheader("After Improvements")
    outcome_box(improved, "Improved")
    mspb_imp = (baseline.get("mspb", 0) or 0) + mspb_delta
    tps_imp = min(100, (baseline.get("tps", 0) or 0) + tps_delta)
    stars_imp = min(5, (baseline.get("stars", 0) or 0) + stars_delta)
    st.metric("MSPB Ratio", f"{mspb_imp:.3f}", delta=f"{mspb_delta:+.3f}" if mspb_delta else None)
    st.metric("VBP TPS", f"{tps_imp:.1f}", delta=f"{tps_delta:+.1f}" if tps_delta else None)
    st.metric("Stars", str(int(stars_imp)), delta=f"{stars_delta:+d}" if stars_delta else None)
    qg2 = improved.get("quality_gate_results", {})
    for gname, gate in qg2.items():
        icon = "✅" if gate["passed"] else "❌"
        st.markdown(f"{icon} {gname}: {gate.get('value', 'N/A')}")

delta_adj = result["delta_adjustment_pct"]
delta_dol = result["delta_dollars"]
st.markdown("---")
d1, d2 = st.columns(2)
d1.metric("Change in Adjustment %", f"{delta_adj:+.3f}%")
d2.metric("Change in Est. Dollars", f"${delta_dol:+,.0f}")

# ── Cluster transition ────────────────────────────────────────────────────────
if "Cluster_Label" in df.columns:
    current_cluster = baseline.get("cluster", "Unknown")
    st.info(f"Current value archetype: **{current_cluster}**")

# ── SHAP waterfall ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("SHAP — Top Improvement Levers")

model, feature_cols, X = load_model()
if not SHAP_AVAILABLE:
    st.info("Install `shap` locally (`pip install shap`) to enable SHAP analysis.")
elif model is not None and X is not None:
    try:
        master_idx = df[df["CCN"] == ccn].index
        if not master_idx.empty and master_idx[0] in X.index:
            row_x = X.loc[master_idx[0]:master_idx[0], feature_cols].fillna(0)
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(row_x)
            shap_series = pd.Series(sv[0], index=feature_cols).abs().sort_values(ascending=False).head(8)
            readable_labels = [feature_label(f) for f in shap_series.index]

            fig = go.Figure(go.Bar(
                x=shap_series.values,
                y=readable_labels,
                orientation="h",
                marker_color=["#1A2B4A"] * len(shap_series),
            ))
            fig.update_layout(title="Top 8 SHAP features (absolute impact on payment adjustment)",
                              xaxis_title="|SHAP value|", height=320, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP analysis not available for this hospital (not in feature matrix).")
    except Exception as e:
        st.warning(f"SHAP analysis unavailable: {e}")
else:
    st.info("Train the ML model first (`python src/model.py`) to enable SHAP analysis.")
