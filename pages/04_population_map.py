"""Page 4: Population Map & Market Analysis."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Population Map", page_icon="🗺️", layout="wide")
st.sidebar.caption("⚠️ For research purposes only. Not financial or medical advice.")

STATE_ABBREV = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA","KS","KY",
    "LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND",
    "OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC",
}


@st.cache_data
def load_master():
    df = pd.read_csv("data/processed/master_hospital.csv", dtype=str)
    df["CCN"] = df["CCN"].str.zfill(6)
    for c in ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating",
              "Payment_Adjustment_Percent", "Composite_Readmission"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Keep only valid US states
    if "State" in df.columns:
        df = df[df["State"].isin(STATE_ABBREV)]
    return df


st.title("🗺️ Population Map & Market Analysis")
st.markdown("Explore hospital performance patterns across the US.")

try:
    df = load_master()
except FileNotFoundError:
    st.error("master_hospital.csv not found. Run `python src/load_data.py` first.")
    st.stop()

name_col = next((c for c in ["Facility Name", "Hospital Name"] if c in df.columns), None)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Filters")
    metric_col = st.selectbox("Color hospitals by", ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating"])
    if "State" in df.columns:
        states = ["All"] + sorted(df["State"].dropna().unique().tolist())
        sel_state = st.selectbox("State", states)
    else:
        sel_state = "All"

    hosp_types = ["All"] + sorted(df["Hospital Type"].dropna().unique().tolist()) if "Hospital Type" in df.columns else ["All"]
    sel_type = st.selectbox("Hospital type", hosp_types)

    ownerships = ["All"] + sorted(df["Hospital Ownership"].dropna().unique().tolist()) if "Hospital Ownership" in df.columns else ["All"]
    sel_own = st.selectbox("Ownership", ownerships)

    star_min = st.slider("Min Star Rating", 1, 5, 1)

    cluster_opts = ["All"] + sorted(df["Cluster_Label"].dropna().unique().tolist()) if "Cluster_Label" in df.columns else ["All"]
    sel_cluster = st.selectbox("Value archetype", cluster_opts)

# ── Apply filters ─────────────────────────────────────────────────────────────
filt = df.copy()
if sel_state != "All" and "State" in filt.columns:
    filt = filt[filt["State"] == sel_state]
if sel_type != "All" and "Hospital Type" in filt.columns:
    filt = filt[filt["Hospital Type"] == sel_type]
if sel_own != "All" and "Hospital Ownership" in filt.columns:
    filt = filt[filt["Hospital Ownership"] == sel_own]
if "Hospital_overall_rating" in filt.columns:
    filt = filt[filt["Hospital_overall_rating"].fillna(0) >= star_min]
if sel_cluster != "All" and "Cluster_Label" in filt.columns:
    filt = filt[filt["Cluster_Label"] == sel_cluster]

st.caption(f"Showing {len(filt):,} hospitals after filters.")

# ── State-level choropleth ────────────────────────────────────────────────────
if "State" in filt.columns:
    state_agg = filt.groupby("State")[metric_col].mean().reset_index()
    state_agg.columns = ["State", "Value"]
    state_agg = state_agg.dropna()

    fig_map = px.choropleth(
        state_agg,
        locations="State",
        locationmode="USA-states",
        color="Value",
        scope="usa",
        color_continuous_scale="RdYlGn_r" if metric_col == "MSPB_Ratio" else "RdYlGn",
        labels={"Value": metric_col.replace("_", " ")},
        title=f"State Average — {metric_col.replace('_', ' ')}",
    )
    fig_map.update_layout(height=480, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

# ── State summary table ───────────────────────────────────────────────────────
if "State" in filt.columns:
    agg_cols = [c for c in ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating"] if c in filt.columns]
    state_summary = filt.groupby("State")[agg_cols].agg(["mean", "count"]).round(3)
    state_summary.columns = [f"{col}_{stat}" for col, stat in state_summary.columns]
    state_summary = state_summary.reset_index().sort_values("MSPB_Ratio_mean")
    st.subheader("State-Level Summary")
    st.dataframe(state_summary, use_container_width=True)

# ── Cluster distribution ──────────────────────────────────────────────────────
if "Cluster_Label" in filt.columns:
    st.subheader("Value Archetype Distribution")
    cluster_counts = filt["Cluster_Label"].value_counts().reset_index()
    cluster_counts.columns = ["Archetype", "Count"]
    fig_pie = px.pie(cluster_counts, names="Archetype", values="Count",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig_pie.update_layout(height=360)
    st.plotly_chart(fig_pie, use_container_width=True)

# ── MSPB vs TPS scatter ───────────────────────────────────────────────────────
scatter_df = filt[["MSPB_Ratio", "Total_Performance_Score"]].dropna()
if name_col:
    scatter_df["Hospital"] = filt.loc[scatter_df.index, name_col]
color_col = "Cluster_Label" if "Cluster_Label" in filt.columns else None
if color_col:
    scatter_df[color_col] = filt.loc[scatter_df.index, color_col]

if not scatter_df.empty:
    st.subheader("Cost vs Quality Scatter")
    fig_sc = px.scatter(
        scatter_df, x="MSPB_Ratio", y="Total_Performance_Score",
        color=color_col, opacity=0.5,
        hover_name="Hospital" if name_col else None,
        labels={"MSPB_Ratio": "MSPB Ratio (cost)", "Total_Performance_Score": "VBP TPS (quality)"},
    )
    fig_sc.add_vline(x=scatter_df["MSPB_Ratio"].median(), line_dash="dash", line_color="gray")
    fig_sc.add_hline(y=scatter_df["Total_Performance_Score"].median(), line_dash="dash", line_color="gray")
    fig_sc.update_layout(height=450)
    st.plotly_chart(fig_sc, use_container_width=True)

# ── CSV download ──────────────────────────────────────────────────────────────
st.subheader("Download Filtered Hospital List")
download_cols = [c for c in [name_col, "CCN", "State", "City", "MSPB_Ratio",
                              "Total_Performance_Score", "Hospital_overall_rating",
                              "Composite_Readmission", "Cluster_Label"] if c and c in filt.columns]
csv = filt[download_cols].to_csv(index=False)
st.download_button("Download CSV", csv, file_name="filtered_hospitals.csv", mime="text/csv")
