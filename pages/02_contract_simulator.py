"""Page 2: VBC Contract Simulator."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import sys
from pathlib import Path

_root = Path(__file__).parent.parent.resolve()
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.simulator import simulate_contract

st.set_page_config(page_title="Contract Simulator", page_icon="📋", layout="wide")
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


st.title("📋 VBC Contract Simulator")
st.markdown("Define contract parameters and instantly project outcomes for any hospital.")

try:
    df = load_master()
except FileNotFoundError:
    st.error("master_hospital.csv not found. Run `python src/load_data.py` first.")
    st.stop()

name_col = next((c for c in ["Facility Name", "Hospital Name"] if c in df.columns), None)
state_col = "State" if "State" in df.columns else None

# ── Layout ────────────────────────────────────────────────────────────────────
left, right = st.columns([1, 2])

with left:
    st.subheader("Hospital Selection")
    if state_col:
        states = ["All"] + sorted(df[state_col].dropna().unique().tolist())
        sel_state = st.selectbox("State", states, key="sim_state")
        filtered = df if sel_state == "All" else df[df[state_col] == sel_state]
    else:
        filtered = df

    search = st.text_input("Search hospital name", key="sim_search")
    if search and name_col:
        filtered = filtered[filtered[name_col].str.contains(search, case=False, na=False)]

    if filtered.empty or name_col is None:
        st.warning("No hospitals found.")
        st.stop()

    selected_name = st.selectbox("Hospital", filtered[name_col].dropna().tolist(), key="sim_hosp")
    row = filtered[filtered[name_col] == selected_name].iloc[0]
    ccn = row["CCN"]

    st.markdown("---")
    st.subheader("Contract Parameters")

    contract_type = st.selectbox("Contract type",
        ["Shared Savings Only", "Shared Savings + Risk", "Capitation"])
    target_mspb = st.slider("Cost target (MSPB ratio)", 0.80, 1.10, 1.00, 0.01,
                            help="Hospitals below this threshold are eligible for savings.")
    min_tps = st.slider("Quality gate — min VBP TPS", 0, 100, 45,
                        help="Minimum Total Performance Score to unlock savings.")
    min_stars = st.slider("Quality gate — min star rating", 1, 5, 3)
    savings_rate = st.slider("Savings share rate", 0.10, 0.80, 0.50, 0.05,
                             help="Fraction of cost savings shared with hospital.")

    show_penalty = contract_type != "Shared Savings Only"
    if show_penalty:
        penalty_threshold = st.slider("Penalty threshold (MSPB)", 1.00, 1.20, 1.05, 0.01)
        penalty_rate = st.slider("Penalty rate", 0.01, 0.10, 0.02, 0.005)
    else:
        penalty_threshold = 999.0
        penalty_rate = 0.0

    benchmark_vol = st.number_input("Est. annual Medicare spend ($)", value=50_000_000, step=1_000_000,
                                    help="Used to convert % adjustment to dollar estimate.")

    benchmark = st.selectbox("Benchmark population",
        ["National Median", "State Median", "Peer Group"])

contract_params = {
    "min_quality_score": float(min_tps),
    "min_stars": int(min_stars),
    "target_mspb": float(target_mspb),
    "penalty_threshold": float(penalty_threshold),
    "savings_share_rate": float(savings_rate),
    "penalty_rate": float(penalty_rate),
    "benchmark_volume": float(benchmark_vol),
}

result = simulate_contract(ccn, contract_params, df)

# ── Results ───────────────────────────────────────────────────────────────────
with right:
    st.subheader(f"Results: {selected_name}")
    mspb_val = result.get("mspb", np.nan)
    tps_val = result.get("tps", np.nan)
    stars_val = result.get("stars", np.nan)

    outcome = result.get("outcome", "unknown")
    adj_pct = result.get("adjustment_pct", 0.0)
    savings_est = result.get("savings_pool_est", 0.0)

    if outcome == "bonus":
        st.success(f"✅ **BONUS** — +{adj_pct:.3f}% adjustment | Est. ${savings_est:,.0f}")
    elif outcome == "penalty":
        st.error(f"❌ **PENALTY** — {adj_pct:.3f}% adjustment | Est. ${abs(savings_est):,.0f} penalty")
    elif outcome == "neutral":
        st.info("➡️ **NEUTRAL** — No bonus or penalty under these contract terms.")
    else:
        st.warning(f"⚠️ Outcome: {outcome}")

    # Quality gate checklist
    st.subheader("Quality Gate Status")
    qg = result.get("quality_gate_results", {})
    for gate_name, gate in qg.items():
        val = gate.get("value", np.nan)
        thresh = gate.get("threshold")
        passed = gate.get("passed", False)
        icon = "✅" if passed else "❌"
        val_str = f"{val:.1f}" if not pd.isna(val) else "N/A"
        st.markdown(f"{icon} **{gate_name}**: {val_str} (threshold: {thresh})")

    gap = result.get("gap_to_qualify")
    if gap:
        st.markdown("**Gap to qualify:**")
        if gap.get("tps_gap", 0) > 0:
            st.markdown(f"- Need +{gap['tps_gap']:.1f} VBP TPS points")
        if gap.get("stars_gap", 0) > 0:
            st.markdown(f"- Need +{gap['stars_gap']} star(s)")

    # Hospital metrics
    st.subheader("Hospital Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("MSPB Ratio", f"{mspb_val:.3f}" if not pd.isna(mspb_val) else "N/A")
    m2.metric("VBP TPS", f"{tps_val:.1f}" if not pd.isna(tps_val) else "N/A")
    m3.metric("Star Rating", f"{int(stars_val)}" if not pd.isna(stars_val) else "N/A")

    # Peer comparison: how many hospitals qualify under these terms?
    st.subheader("Peer Comparison")
    if state_col and "State" in row.index:
        peer_scope = df[df["State"] == row["State"]] if benchmark == "State Median" else df
    else:
        peer_scope = df

    total_peers = len(peer_scope.dropna(subset=["MSPB_Ratio", "Total_Performance_Score"]))
    qualifying = sum(
        1 for _, pr in peer_scope.iterrows()
        if not pd.isna(pr.get("MSPB_Ratio")) and not pd.isna(pr.get("Total_Performance_Score"))
        and pr["MSPB_Ratio"] < target_mspb
        and pr["Total_Performance_Score"] >= min_tps
    )
    pct_qualify = qualifying / total_peers * 100 if total_peers > 0 else 0

    fig = go.Figure(go.Bar(
        x=["Would Qualify", "Would Not Qualify"],
        y=[qualifying, total_peers - qualifying],
        marker_color=["#2E8B57", "#CC3333"],
    ))
    fig.update_layout(title=f"{benchmark}: {qualifying}/{total_peers} hospitals qualify ({pct_qualify:.1f}%)",
                      height=280, margin=dict(t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
