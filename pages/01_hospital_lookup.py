"""Page 1: Hospital Lookup & Value Profile."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path

_root = Path(__file__).parent.parent.resolve()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.hcahps_labels import label as hcahps_label

st.set_page_config(page_title="Hospital Lookup", page_icon="🔍", layout="wide")
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


def traffic_light(value, low=0.95, high=1.05):
    if pd.isna(value):
        return "⚪"
    if value < low:
        return "🟢"
    elif value <= high:
        return "🟡"
    return "🔴"


def star_display(stars):
    if pd.isna(stars):
        return "N/A"
    n = int(stars)
    return "⭐" * n + "☆" * (5 - n)


st.title("🔍 Hospital Lookup & Value Profile")

try:
    df = load_master()
except FileNotFoundError:
    st.error("master_hospital.csv not found. Run `python src/load_data.py` first.")
    st.stop()

name_col = next((c for c in ["Facility Name", "Hospital Name"] if c in df.columns), None)
state_col = "State" if "State" in df.columns else None

col1, col2 = st.columns([3, 1])
with col1:
    search = st.text_input("Search by hospital name", placeholder="e.g. Mayo Clinic")
with col2:
    states = ["All"] + sorted(df[state_col].dropna().unique().tolist()) if state_col else ["All"]
    selected_state = st.selectbox("Filter by state", states)

filtered = df.copy()
if selected_state != "All" and state_col:
    filtered = filtered[filtered[state_col] == selected_state]
if search and name_col:
    filtered = filtered[filtered[name_col].str.contains(search, case=False, na=False)]

if filtered.empty:
    st.warning("No hospitals match your search.")
    st.stop()

if name_col:
    hospital_names = filtered[name_col].dropna().tolist()
    selected_name = st.selectbox("Select hospital", hospital_names)
    row = filtered[filtered[name_col] == selected_name].iloc[0]
else:
    row = filtered.iloc[0]

st.markdown("---")

# ── Value Profile Card ────────────────────────────────────────────────────────
hosp_name = row.get(name_col, "Unknown") if name_col else "Unknown"
city = row.get("City", "")
state = row.get("State", "")
location = f"{city}, {state}".strip(", ")

st.subheader(f"🏥 {hosp_name}")
st.caption(f"{location} | CCN: {row['CCN']}")

if "Cluster_Label" in row and pd.notna(row["Cluster_Label"]):
    cluster_colors = {
        "High Value": "green", "Cost Efficient / Lower Quality": "blue",
        "High Quality / Higher Cost": "orange", "Average Performers": "gray",
        "Low Value": "red", "Emerging Performers": "purple",
    }
    cluster = row["Cluster_Label"]
    color = cluster_colors.get(cluster, "gray")
    st.markdown(f"**Value Archetype:** :{color}[{cluster}]")

c1, c2, c3, c4 = st.columns(4)

mspb = row.get("MSPB_Ratio", np.nan)
tps = row.get("Total_Performance_Score", np.nan)
stars = row.get("Hospital_overall_rating", np.nan)
composite_readmit = row.get("Composite_Readmission", np.nan)

with c1:
    indicator = traffic_light(mspb)
    mspb_str = f"{mspb:.3f}" if not pd.isna(mspb) else "N/A"
    st.metric(f"{indicator} MSPB Ratio", mspb_str,
              help="Episode cost vs national median. <1.0 = more efficient.")

with c2:
    tps_str = f"{tps:.1f} / 100" if not pd.isna(tps) else "N/A"
    st.metric("📊 VBP Total Performance Score", tps_str,
              help="CMS quality composite score (0–100).")

with c3:
    st.metric("⭐ Star Rating", star_display(stars))

with c4:
    cr_str = f"{composite_readmit:.3f}" if not pd.isna(composite_readmit) else "N/A"
    st.metric("🔄 Composite Readmission Ratio", cr_str,
              help="Mean excess readmission ratio across conditions. >1.0 = worse than expected.")

# ── VBP Domain breakdown ──────────────────────────────────────────────────────
domain_cols = {
    "Domain_Safety_Score": "Safety",
    "Domain_Clinical_Score": "Clinical Outcomes",
    "Domain_Efficiency_Score": "Efficiency & Cost",
    "Domain_Engagement_Score": "Person & Community",
}
domain_data = {label: pd.to_numeric(row.get(col, np.nan), errors="coerce")
               for col, label in domain_cols.items() if col in row.index}
domain_data = {k: v for k, v in domain_data.items() if not pd.isna(v)}

if domain_data:
    st.subheader("VBP Domain Scores")
    fig = go.Figure(go.Bar(
        x=list(domain_data.values()),
        y=list(domain_data.keys()),
        orientation="h",
        marker_color=["#1A2B4A"] * len(domain_data),
    ))
    fig.update_layout(xaxis_title="Score", xaxis_range=[0, 100], height=220,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

# ── HCAHPS highlights ─────────────────────────────────────────────────────────
hcahps_cols = [c for c in df.columns if c.startswith("HCAHPS_")]
hcahps_vals = {
    hcahps_label(c.replace("HCAHPS_", "")): pd.to_numeric(row.get(c, np.nan), errors="coerce")
    for c in hcahps_cols
}
hcahps_vals = {k: v for k, v in hcahps_vals.items() if not pd.isna(v)}
if hcahps_vals:
    st.subheader("HCAHPS Patient Experience Highlights")
    hcahps_df = pd.DataFrame(list(hcahps_vals.items()), columns=["Measure", "Score (%)"])
    hcahps_df = hcahps_df.sort_values("Score (%)", ascending=False)
    st.dataframe(hcahps_df, use_container_width=True, hide_index=True)

# ── Value quadrant scatter ────────────────────────────────────────────────────
st.subheader("Value Quadrant — Cost vs Quality")
quad_df = df[["MSPB_Ratio", "Total_Performance_Score"]].copy()
if name_col:
    quad_df["name"] = df[name_col]
quad_df = quad_df.dropna()
fig2 = px.scatter(quad_df, x="MSPB_Ratio", y="Total_Performance_Score",
                  opacity=0.3, color_discrete_sequence=["#AAAAAA"],
                  labels={"MSPB_Ratio": "MSPB Ratio (cost)", "Total_Performance_Score": "VBP TPS (quality)"})
if not pd.isna(mspb) and not pd.isna(tps):
    fig2.add_scatter(x=[mspb], y=[tps], mode="markers",
                     marker=dict(size=14, color="#E63946", symbol="star"),
                     name=hosp_name)
fig2.add_vline(x=quad_df["MSPB_Ratio"].median(), line_dash="dash", line_color="gray")
fig2.add_hline(y=quad_df["Total_Performance_Score"].median(), line_dash="dash", line_color="gray")
fig2.update_layout(height=400, showlegend=True)
st.plotly_chart(fig2, use_container_width=True)

# ── Peer comparison in state ──────────────────────────────────────────────────
if state_col:
    state_val = row.get("State", None)
    if state_val and not pd.isna(state_val):
        peers = df[df["State"] == state_val][["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating"]].dropna()
        st.subheader(f"Peer Comparison — {state_val}")
        pc1, pc2, pc3 = st.columns(3)
        if not pd.isna(mspb):
            pct = (peers["MSPB_Ratio"] > mspb).mean() * 100
            pc1.metric("Better MSPB than", f"{pct:.0f}% of state peers")
        if not pd.isna(tps):
            pct = (peers["Total_Performance_Score"] < tps).mean() * 100
            pc2.metric("Higher TPS than", f"{pct:.0f}% of state peers")
        if not pd.isna(stars):
            pct = (peers["Hospital_overall_rating"] <= stars).mean() * 100
            pc3.metric("Star rating ≥", f"{pct:.0f}% of state peers")
