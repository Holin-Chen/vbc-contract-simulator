"""Phase 1: Load and join all 5 CMS datasets on Facility ID."""

import pandas as pd
import numpy as np
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")

# Readmission measure IDs to pivot
READMIT_IDS = [
    "READM_30_AMI", "READM_30_CABG", "READM_30_COPD",
    "READM_30_HF", "READM_30_HIP_KNEE", "READM_30_PN",
    "EDAC_30_AMI", "EDAC_30_HF", "EDAC_30_PN",
]


def _normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.zfill(6)


def load_general_info() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Hospital_General_Information.csv", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Facility ID": "CCN",
        "City/Town": "City",
        "Hospital overall rating": "Hospital_overall_rating",
    })
    df["CCN"] = _normalize_id(df["CCN"])
    return df


def load_mspb() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Medicare_Spending_Per_Beneficiary_Hospital.csv", dtype=str)
    df.columns = df.columns.str.strip()
    # Long format: one row per (facility, measure). Only one measure: MSPB-1
    df = df[df["Measure ID"] == "MSPB-1"].copy()
    df = df.rename(columns={"Facility ID": "CCN", "Score": "MSPB_Ratio"})
    df["CCN"] = _normalize_id(df["CCN"])
    return df[["CCN", "MSPB_Ratio"]]


def load_vbp() -> pd.DataFrame:
    df = pd.read_csv(RAW / "VBP_Hospital_TPS.csv", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Facility ID": "CCN",
        "Total Performance Score": "Total_Performance_Score",
        "Weighted Normalized Clinical Outcomes Domain Score": "Domain_Clinical_Score",
        "Weighted Safety Domain Score": "Domain_Safety_Score",
        "Weighted Person And Community Engagement Domain Score": "Domain_Engagement_Score",
        "Weighted Efficiency And Cost Reduction Domain Score": "Domain_Efficiency_Score",
    })
    df["CCN"] = _normalize_id(df["CCN"])
    keep = ["CCN", "Total_Performance_Score", "Domain_Clinical_Score",
            "Domain_Safety_Score", "Domain_Engagement_Score", "Domain_Efficiency_Score"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def load_hcahps() -> pd.DataFrame:
    df = pd.read_csv(RAW / "HCAHPS_Hospital.csv", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Facility ID": "CCN"})
    df["CCN"] = _normalize_id(df["CCN"])

    # Pivot: one column per HCAHPS Measure ID, value = HCAHPS Answer Percent
    df_pivot = df.pivot_table(
        index="CCN",
        columns="HCAHPS Measure ID",
        values="HCAHPS Answer Percent",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns = ["CCN"] + [f"HCAHPS_{c}" for c in df_pivot.columns[1:]]
    return df_pivot


def load_readmissions() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Unplanned_Hospital_Visits_Provider_Data.csv", dtype=str)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Facility ID": "CCN"})
    df["CCN"] = _normalize_id(df["CCN"])

    # Keep only readmission/EDAC measures
    df = df[df["Measure ID"].isin(READMIT_IDS)].copy()
    df_pivot = df.pivot_table(
        index="CCN",
        columns="Measure ID",
        values="Score",
        aggfunc="first",
    ).reset_index()
    df_pivot.columns = ["CCN"] + [f"Readmission_{c}" for c in df_pivot.columns[1:]]
    return df_pivot


def _derive_payment_adjustment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate VBP payment adjustment from TPS percentile rank.
    CMS adjusts payments from -2% (lowest TPS) to +2% (highest TPS).
    """
    tps = pd.to_numeric(df["Total_Performance_Score"], errors="coerce")
    pct_rank = tps.rank(pct=True, na_option="keep")
    # Scale 0–1 percentile rank to -2% to +2%
    df["Payment_Adjustment_Percent"] = (pct_rank * 4.0 - 2.0).round(4)
    return df


def build_master() -> pd.DataFrame:
    print("Loading datasets...")
    general = load_general_info()
    mspb = load_mspb()
    vbp = load_vbp()
    hcahps = load_hcahps()
    readmissions = load_readmissions()

    print(f"  General info:  {len(general):>5} rows")
    print(f"  MSPB:          {len(mspb):>5} rows")
    print(f"  VBP TPS:       {len(vbp):>5} rows")
    print(f"  HCAHPS:        {len(hcahps):>5} rows")
    print(f"  Readmissions:  {len(readmissions):>5} rows")

    master = general.copy()
    for df, name in [(mspb, "MSPB"), (vbp, "VBP"), (hcahps, "HCAHPS"), (readmissions, "Readmissions")]:
        master = master.merge(df, on="CCN", how="left")
        print(f"  After joining {name}: {len(master)} rows, {len(master.columns)} cols")

    # Derive payment adjustment from TPS percentile
    master = _derive_payment_adjustment(master)

    # Standardize missing value strings to NaN
    master = master.replace(["Not Available", "N/A", "Not Applicable", ""], np.nan)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    master.to_csv(PROCESSED / "master_hospital.csv", index=False)
    print(f"\nSaved master_hospital.csv: {len(master)} rows, {len(master.columns)} columns")

    key_cols = ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating",
                "Payment_Adjustment_Percent"]
    print("\nMissing value % for key columns:")
    for c in key_cols:
        if c in master.columns:
            pct = master[c].isna().mean() * 100
            print(f"  {c}: {pct:.1f}%")

    return master


if __name__ == "__main__":
    build_master()
