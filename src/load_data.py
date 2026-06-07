"""Phase 1: Load and join all 5 CMS datasets on CCN (Provider ID)."""

import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROCESSED = Path("data/processed")


def load_general_info() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Hospital_General_Information.csv", dtype=str)
    df.columns = df.columns.str.strip()
    # Normalize join key
    df = df.rename(columns={"Provider ID": "CCN"})
    df["CCN"] = df["CCN"].str.strip().str.zfill(6)
    return df


def load_mspb() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Medicare_Spending_Per_Beneficiary_Hospital.csv", dtype=str)
    df.columns = df.columns.str.strip()
    key_col = next(c for c in df.columns if "Facility ID" in c or "Provider ID" in c)
    df = df.rename(columns={key_col: "CCN"})
    df["CCN"] = df["CCN"].str.strip().str.zfill(6)
    # Keep relevant columns
    score_col = next((c for c in df.columns if "MSPB" in c and "Score" in c), None)
    ratio_col = next((c for c in df.columns if "Ratio" in c), None)
    cols = ["CCN"]
    if ratio_col:
        cols.append(ratio_col)
        df = df.rename(columns={ratio_col: "MSPB_Ratio"})
    if score_col:
        cols_extra = [score_col]
        df = df.rename(columns={score_col: "MSPB_Score"})
        cols += ["MSPB_Score"]
    return df[cols] if all(c in df.columns for c in cols) else df[["CCN", "MSPB_Ratio"]] if "MSPB_Ratio" in df.columns else df[["CCN"]]


def load_vbp() -> pd.DataFrame:
    df = pd.read_csv(RAW / "VBP_Hospital_TPS.csv", dtype=str)
    df.columns = df.columns.str.strip()
    key_col = next(c for c in df.columns if "Facility ID" in c or "Provider ID" in c or "CCN" in c)
    df = df.rename(columns={key_col: "CCN"})
    df["CCN"] = df["CCN"].str.strip().str.zfill(6)
    rename_map = {}
    for c in df.columns:
        if "Total Performance Score" in c or "Total_Performance_Score" in c:
            rename_map[c] = "Total_Performance_Score"
        elif "Safety" in c and "Score" in c:
            rename_map[c] = "Domain_Safety_Score"
        elif "Clinical" in c and "Score" in c:
            rename_map[c] = "Domain_Clinical_Score"
        elif "Efficiency" in c and "Score" in c:
            rename_map[c] = "Domain_Efficiency_Score"
        elif "Engagement" in c and "Score" in c or "Person" in c and "Score" in c:
            rename_map[c] = "Domain_Engagement_Score"
        elif "Payment" in c and "Adjustment" in c:
            rename_map[c] = "Payment_Adjustment_Percent"
    df = df.rename(columns=rename_map)
    keep = ["CCN"] + [v for v in rename_map.values() if v in df.columns]
    return df[keep]


def load_hcahps() -> pd.DataFrame:
    df = pd.read_csv(RAW / "HCAHPS_Hospital.csv", dtype=str)
    df.columns = df.columns.str.strip()
    key_col = next(c for c in df.columns if "Facility ID" in c or "Provider ID" in c)
    df = df.rename(columns={key_col: "CCN"})
    df["CCN"] = df["CCN"].str.strip().str.zfill(6)

    measure_col = next((c for c in df.columns if "HCAHPS Measure ID" in c or "Measure ID" in c), None)
    answer_col = next((c for c in df.columns if "Answer Percent" in c or "HCAHPS Answer Percent" in c), None)

    if measure_col and answer_col:
        df_pivot = df.pivot_table(
            index="CCN",
            columns=measure_col,
            values=answer_col,
            aggfunc="first"
        ).reset_index()
        df_pivot.columns = ["CCN"] + [f"HCAHPS_{c}" for c in df_pivot.columns[1:]]
        return df_pivot
    return df[["CCN"]].drop_duplicates()


def load_readmissions() -> pd.DataFrame:
    df = pd.read_csv(RAW / "Unplanned_Hospital_Visits_Provider_Data.csv", dtype=str)
    df.columns = df.columns.str.strip()
    key_col = next(c for c in df.columns if "Facility ID" in c or "Provider ID" in c)
    df = df.rename(columns={key_col: "CCN"})
    df["CCN"] = df["CCN"].str.strip().str.zfill(6)

    measure_col = next((c for c in df.columns if "Measure ID" in c), None)
    score_col = next((c for c in df.columns if "Score" in c or "Ratio" in c), None)

    if measure_col and score_col:
        # Filter to readmission measures only
        readmission_df = df[df[measure_col].str.contains("READM|EDAC", na=False)]
        if readmission_df.empty:
            readmission_df = df
        df_pivot = readmission_df.pivot_table(
            index="CCN",
            columns=measure_col,
            values=score_col,
            aggfunc="first"
        ).reset_index()
        df_pivot.columns = ["CCN"] + [f"Readmission_{c}" for c in df_pivot.columns[1:]]
        return df_pivot
    return df[["CCN"]].drop_duplicates()


def build_master() -> pd.DataFrame:
    print("Loading datasets...")
    general = load_general_info()
    mspb = load_mspb()
    vbp = load_vbp()
    hcahps = load_hcahps()
    readmissions = load_readmissions()

    print(f"  General info: {len(general)} rows")
    print(f"  MSPB: {len(mspb)} rows")
    print(f"  VBP TPS: {len(vbp)} rows")
    print(f"  HCAHPS: {len(hcahps)} rows")
    print(f"  Readmissions: {len(readmissions)} rows")

    master = general.copy()
    for df, name in [(mspb, "MSPB"), (vbp, "VBP"), (hcahps, "HCAHPS"), (readmissions, "Readmissions")]:
        master = master.merge(df, on="CCN", how="left")
        print(f"  After joining {name}: {len(master)} rows, {len(master.columns)} cols")

    # Fill missing values
    num_cols = master.select_dtypes(include="number").columns
    master[num_cols] = master[num_cols].fillna(float("nan"))
    str_cols = master.select_dtypes(include="object").columns
    for col in str_cols:
        master[col] = master[col].replace(["Not Available", "N/A", "Not Applicable", ""], float("nan"))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    master.to_csv(PROCESSED / "master_hospital.csv", index=False)
    print(f"\nSaved master_hospital.csv: {len(master)} rows, {len(master.columns)} columns")

    # Missing value report for key columns
    key_cols = [c for c in ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating",
                             "Payment_Adjustment_Percent"] if c in master.columns]
    print("\nMissing value % for key columns:")
    for c in key_cols:
        pct = master[c].isna().mean() * 100
        print(f"  {c}: {pct:.1f}%")

    return master


if __name__ == "__main__":
    build_master()
