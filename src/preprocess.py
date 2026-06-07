"""Phase 2: Feature engineering, normalization, and EDA."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

PROCESSED = Path("data/processed")


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_features(master: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = master.copy()

    # 1. Parse key numeric columns
    df["MSPB_Ratio"] = _to_float(df.get("MSPB_Ratio"))
    df["Total_Performance_Score"] = _to_float(df.get("Total_Performance_Score"))
    df["Hospital_overall_rating"] = _to_float(df.get("Hospital_overall_rating"))
    df["Payment_Adjustment_Percent"] = _to_float(df.get("Payment_Adjustment_Percent"))

    # 2. HCAHPS columns — already pivoted wide; coerce to float
    hcahps_cols = [c for c in df.columns if c.startswith("HCAHPS_")]
    for c in hcahps_cols:
        df[c] = _to_float(df[c])

    # 3. Readmission columns — pivot already wide; coerce and build composite
    readmit_cols = [c for c in df.columns if c.startswith("Readmission_")]
    for c in readmit_cols:
        df[c] = _to_float(df[c])
    if readmit_cols:
        df["Composite_Readmission"] = df[readmit_cols].mean(axis=1)
    else:
        df["Composite_Readmission"] = np.nan

    # 4. VBP domain scores
    domain_cols = [c for c in df.columns if c.startswith("Domain_")]
    for c in domain_cols:
        df[c] = _to_float(df[c])

    # 5. Value quadrant label (4 categories from MSPB / TPS median split)
    mspb_med = df["MSPB_Ratio"].median()
    tps_med = df["Total_Performance_Score"].median()

    def quadrant(row):
        if pd.isna(row["MSPB_Ratio"]) or pd.isna(row["Total_Performance_Score"]):
            return "Unknown"
        low_cost = row["MSPB_Ratio"] <= mspb_med
        high_qual = row["Total_Performance_Score"] >= tps_med
        if low_cost and high_qual:
            return "High Value"
        elif low_cost:
            return "Cost Efficient"
        elif high_qual:
            return "High Quality"
        else:
            return "Low Value"

    df["Value_Quadrant"] = df.apply(quadrant, axis=1)

    # 6. One-hot encode categorical columns
    cat_cols = [c for c in ["Hospital Type", "Hospital Ownership", "State"] if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)

    # 7. Build feature matrix X
    numeric_feature_cols = (
        ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating", "Composite_Readmission"]
        + domain_cols
        + hcahps_cols
        + [c for c in df.columns if c.startswith("Hospital Type_") or c.startswith("Hospital Ownership_") or c.startswith("State_")]
    )
    numeric_feature_cols = [c for c in numeric_feature_cols if c in df.columns]

    X = df[numeric_feature_cols].copy()
    y = df["Payment_Adjustment_Percent"].copy()

    # 8. Normalize numeric features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X.fillna(0)), columns=X.columns, index=X.index)

    # Save
    PROCESSED.mkdir(parents=True, exist_ok=True)
    joblib.dump({"X": X_scaled, "X_raw": X, "y": y, "scaler": scaler, "feature_cols": numeric_feature_cols}, PROCESSED / "features.pkl")
    print(f"Saved features.pkl: X shape={X_scaled.shape}, y non-null={y.notna().sum()}")

    # 9. EDA
    print("\nTop 10 feature correlations with Payment_Adjustment_Percent:")
    corr_df = X_scaled.copy()
    corr_df["target"] = y.values
    corr = corr_df.corr()["target"].drop("target").abs().sort_values(ascending=False).head(10)
    print(corr.to_string())

    return X_scaled, y, df


if __name__ == "__main__":
    master = pd.read_csv(PROCESSED / "master_hospital.csv", dtype=str)
    X, y, df_full = build_features(master)
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target distribution:\n{y.describe()}")
