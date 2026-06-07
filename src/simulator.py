"""Phase 4: Deterministic VBC contract simulation engine."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

PROCESSED = Path("data/processed")


def _load_master() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / "master_hospital.csv", dtype=str)
    df["CCN"] = df["CCN"].str.zfill(6)
    numeric_cols = ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating",
                    "Payment_Adjustment_Percent", "Composite_Readmission"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def simulate_contract(ccn: str, contract_params: dict[str, Any], df: pd.DataFrame | None = None) -> dict:
    """
    Simulate VBC contract outcome for a hospital.

    contract_params keys:
      - min_quality_score: float  (VBP TPS gate, e.g. 45)
      - min_stars: int            (star rating gate, e.g. 3)
      - target_mspb: float        (cost target, e.g. 0.95)
      - penalty_threshold: float  (MSPB above which penalty applies, e.g. 1.05)
      - savings_share_rate: float (fraction of savings shared, e.g. 0.5)
      - penalty_rate: float       (fraction of excess cost as penalty, e.g. 0.02)
      - benchmark_volume: float   (estimated annual Medicare spend for $ conversion, e.g. 50_000_000)
    """
    if df is None:
        df = _load_master()

    ccn = str(ccn).zfill(6)
    row = df[df["CCN"] == ccn]
    if row.empty:
        return {"error": f"Hospital CCN {ccn} not found", "outcome": "error"}

    row = row.iloc[0]
    mspb = row.get("MSPB_Ratio", np.nan)
    tps = row.get("Total_Performance_Score", np.nan)
    stars = row.get("Hospital_overall_rating", np.nan)

    min_tps = contract_params.get("min_quality_score", 45.0)
    min_stars = contract_params.get("min_stars", 3)
    target_mspb = contract_params.get("target_mspb", 1.0)
    penalty_threshold = contract_params.get("penalty_threshold", 1.05)
    savings_rate = contract_params.get("savings_share_rate", 0.5)
    penalty_rate = contract_params.get("penalty_rate", 0.02)
    benchmark_vol = contract_params.get("benchmark_volume", 50_000_000)

    # Quality gate checks
    tps_pass = bool(not pd.isna(tps) and tps >= min_tps)
    star_pass = bool(not pd.isna(stars) and int(stars) >= min_stars)
    quality_gate_passed = tps_pass and star_pass

    quality_gate_results = {
        "VBP TPS": {"value": tps, "threshold": min_tps, "passed": tps_pass},
        "Star Rating": {"value": stars, "threshold": min_stars, "passed": star_pass},
    }

    if pd.isna(mspb):
        return {
            "outcome": "insufficient_data",
            "adjustment_pct": 0.0,
            "savings_pool_est": 0.0,
            "quality_gate_results": quality_gate_results,
            "gap_to_qualify": None,
            "hospital_name": row.get("Facility Name", "Unknown"),
            "mspb": mspb,
            "tps": tps,
            "stars": stars,
        }

    cost_ratio = mspb / target_mspb
    cost_savings_pct = max(0.0, 1.0 - cost_ratio)

    gap_to_qualify = None
    if quality_gate_passed and cost_ratio < 1.0:
        bonus_pct = cost_savings_pct * savings_rate
        savings_pool = cost_savings_pct * benchmark_vol
        bonus_dollars = savings_pool * savings_rate
        outcome = "bonus"
        adjustment_pct = bonus_pct
        savings_pool_est = bonus_dollars
    elif mspb > penalty_threshold:
        penalty_pct = (mspb - penalty_threshold) * penalty_rate
        penalty_dollars = penalty_pct * benchmark_vol
        outcome = "penalty"
        adjustment_pct = -penalty_pct
        savings_pool_est = -penalty_dollars
        if not quality_gate_passed:
            gap_to_qualify = {
                "tps_gap": max(0, min_tps - (tps or 0)),
                "stars_gap": max(0, min_stars - int(stars or 0)),
            }
    else:
        outcome = "neutral"
        adjustment_pct = 0.0
        savings_pool_est = 0.0
        if not quality_gate_passed:
            gap_to_qualify = {
                "tps_gap": max(0, min_tps - (tps or 0)),
                "stars_gap": max(0, min_stars - int(stars or 0)),
            }

    return {
        "outcome": outcome,
        "adjustment_pct": round(adjustment_pct * 100, 4),  # as percentage
        "savings_pool_est": round(savings_pool_est, 2),
        "quality_gate_results": quality_gate_results,
        "quality_gate_passed": quality_gate_passed,
        "gap_to_qualify": gap_to_qualify,
        "hospital_name": row.get("Facility Name", "Unknown"),
        "mspb": mspb,
        "tps": tps,
        "stars": stars,
        "cluster": row.get("Cluster_Label", "Unknown"),
    }


def what_if_simulate(ccn: str, improvements: dict[str, float],
                     contract_params: dict[str, Any], df: pd.DataFrame | None = None) -> dict:
    """
    Simulate contract outcome after hypothetical metric improvements.

    improvements keys (additive deltas):
      - mspb: e.g. -0.05 (reduce MSPB ratio by 0.05)
      - tps: e.g. +5 (raise VBP TPS by 5 points)
      - stars: e.g. +1 (raise star rating by 1)
      - readmission: e.g. -0.05 (reduce composite readmission ratio)
    """
    if df is None:
        df = _load_master()

    df_mod = df.copy()
    ccn_z = str(ccn).zfill(6)
    idx = df_mod[df_mod["CCN"] == ccn_z].index
    if idx.empty:
        return {"error": f"Hospital CCN {ccn} not found"}

    if "mspb" in improvements:
        df_mod.loc[idx, "MSPB_Ratio"] = df_mod.loc[idx, "MSPB_Ratio"] + improvements["mspb"]
    if "tps" in improvements:
        df_mod.loc[idx, "Total_Performance_Score"] = (
            df_mod.loc[idx, "Total_Performance_Score"] + improvements["tps"]
        ).clip(upper=100)
    if "stars" in improvements:
        df_mod.loc[idx, "Hospital_overall_rating"] = (
            df_mod.loc[idx, "Hospital_overall_rating"] + improvements["stars"]
        ).clip(upper=5)

    baseline = simulate_contract(ccn, contract_params, df)
    improved = simulate_contract(ccn, contract_params, df_mod)

    delta_pct = improved["adjustment_pct"] - baseline["adjustment_pct"]
    delta_dollars = improved["savings_pool_est"] - baseline["savings_pool_est"]

    return {
        "baseline": baseline,
        "improved": improved,
        "delta_adjustment_pct": round(delta_pct, 4),
        "delta_dollars": round(delta_dollars, 2),
        "improvements_applied": improvements,
    }


# ── Unit tests ────────────────────────────────────────────────────────────────

def _make_test_df(**overrides) -> pd.DataFrame:
    base = {
        "CCN": "000001",
        "Facility Name": "Test Hospital",
        "MSPB_Ratio": 1.0,
        "Total_Performance_Score": 50.0,
        "Hospital_overall_rating": 4.0,
        "Composite_Readmission": 1.0,
        "Cluster_Label": "Average Performers",
    }
    base.update(overrides)
    for k, v in base.items():
        if k not in ("CCN", "Facility Name", "Cluster_Label"):
            base[k] = float(v)
    return pd.DataFrame([base])


DEFAULT_PARAMS = {
    "min_quality_score": 45.0,
    "min_stars": 3,
    "target_mspb": 1.0,
    "penalty_threshold": 1.05,
    "savings_share_rate": 0.5,
    "penalty_rate": 0.02,
    "benchmark_volume": 50_000_000,
}


def test_clear_bonus():
    df = _make_test_df(MSPB_Ratio=0.90, Total_Performance_Score=60.0, Hospital_overall_rating=4.0)
    result = simulate_contract("000001", DEFAULT_PARAMS, df)
    assert result["outcome"] == "bonus", f"Expected bonus, got {result['outcome']}"
    assert result["adjustment_pct"] > 0
    print("PASS test_clear_bonus")


def test_clear_penalty():
    df = _make_test_df(MSPB_Ratio=1.10, Total_Performance_Score=60.0, Hospital_overall_rating=4.0)
    result = simulate_contract("000001", DEFAULT_PARAMS, df)
    assert result["outcome"] == "penalty", f"Expected penalty, got {result['outcome']}"
    assert result["adjustment_pct"] < 0
    print("PASS test_clear_penalty")


def test_quality_gate_fail():
    df = _make_test_df(MSPB_Ratio=0.90, Total_Performance_Score=30.0, Hospital_overall_rating=2.0)
    result = simulate_contract("000001", DEFAULT_PARAMS, df)
    assert result["outcome"] == "neutral", f"Expected neutral (quality gate fail), got {result['outcome']}"
    assert result["quality_gate_passed"] is False
    print("PASS test_quality_gate_fail")


def test_neutral_zone():
    df = _make_test_df(MSPB_Ratio=1.02, Total_Performance_Score=60.0, Hospital_overall_rating=4.0)
    result = simulate_contract("000001", DEFAULT_PARAMS, df)
    assert result["outcome"] == "neutral", f"Expected neutral, got {result['outcome']}"
    print("PASS test_neutral_zone")


def test_what_if_crosses_threshold():
    df = _make_test_df(MSPB_Ratio=1.02, Total_Performance_Score=40.0, Hospital_overall_rating=3.0)
    improvements = {"mspb": -0.08, "tps": 10.0}
    result = what_if_simulate("000001", improvements, DEFAULT_PARAMS, df)
    assert result["baseline"]["outcome"] in ("neutral", "penalty")
    assert result["improved"]["outcome"] == "bonus", f"Expected bonus after improvement, got {result['improved']['outcome']}"
    assert result["delta_adjustment_pct"] > 0
    print("PASS test_what_if_crosses_threshold")


def run_tests():
    test_clear_bonus()
    test_clear_penalty()
    test_quality_gate_fail()
    test_neutral_zone()
    test_what_if_crosses_threshold()
    print("\nAll 5 tests passed.")


if __name__ == "__main__":
    run_tests()
