"""Phase 3: XGBoost payment predictor + KMeans clustering + SHAP explainability."""

import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, silhouette_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, XGBRegressor

PROCESSED = Path("data/processed")
MODELS = Path("models")
CLUSTER_FEATURES = ["MSPB_Ratio", "Total_Performance_Score", "Hospital_overall_rating", "Composite_Readmission"]

CLUSTER_ARCHETYPES = {
    0: "High Value",
    1: "Cost Efficient / Lower Quality",
    2: "High Quality / Higher Cost",
    3: "Average Performers",
    4: "Low Value",
    5: "Emerging Performers",
}


def _load_features():
    data = joblib.load(PROCESSED / "features.pkl")
    return data["X"], data["X_raw"], data["y"], data["scaler"], data["feature_cols"]


def train_payment_predictor():
    X, X_raw, y, scaler, feature_cols = _load_features()

    # Drop rows where target is null
    mask = y.notna()
    X_fit = X[mask]
    y_fit = y[mask]

    print(f"Training on {mask.sum()} hospitals with Payment_Adjustment_Percent")

    # Sample weights to amplify outliers (|adj| > 0.5%)
    weights = np.where(y_fit.abs() > 0.5, 3.0, 1.0)

    # Regressor
    reg = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    cv_rmse = cross_val_score(reg, X_fit, y_fit, cv=5, scoring="neg_root_mean_squared_error")
    reg.fit(X_fit, y_fit, sample_weight=weights)
    preds = reg.predict(X_fit)
    rmse = mean_squared_error(y_fit, preds) ** 0.5
    r2 = r2_score(y_fit, preds)
    print(f"Regressor — CV RMSE: {-cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} | Train RMSE: {rmse:.4f} | R²: {r2:.4f}")

    # Classifier: bonus=1, penalty=-1, neutral=0
    y_cls = np.where(y_fit > 0.5, 1, np.where(y_fit < -0.5, -1, 0))
    # Map to 0,1,2 for XGB
    y_cls_mapped = y_cls + 1  # -1→0, 0→1, 1→2
    clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                        use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    clf.fit(X_fit, y_cls_mapped)
    proba = clf.predict_proba(X_fit)
    auc = roc_auc_score(pd.get_dummies(y_cls_mapped).values, proba, multi_class="ovr")
    print(f"Classifier — ROC-AUC (OvR): {auc:.4f}")

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"regressor": reg, "classifier": clf, "feature_cols": feature_cols}, MODELS / "xgb_payment_adj.pkl")
    print("Saved models/xgb_payment_adj.pkl")
    return reg, clf


def train_cluster_model():
    _, X_raw, _, _, _ = _load_features()
    master = pd.read_csv(PROCESSED / "master_hospital.csv", dtype=str)

    # Use raw (unscaled) cluster features
    cluster_df = X_raw[CLUSTER_FEATURES].copy()
    # Fill with column median for clustering
    cluster_df = cluster_df.fillna(cluster_df.median())

    best_k, best_score = 4, -1
    for k in range(4, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(cluster_df)
        score = silhouette_score(cluster_df, labels)
        print(f"  k={k}: silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k = score, k

    print(f"Best k={best_k} (silhouette={best_score:.4f})")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(cluster_df)

    archetype_map = {i: CLUSTER_ARCHETYPES.get(i, f"Cluster {i}") for i in range(best_k)}

    # Assign archetypes by cluster centroid characteristics
    centroids = pd.DataFrame(km.cluster_centers_, columns=CLUSTER_FEATURES)
    centroids["cluster"] = range(best_k)
    centroids_sorted = centroids.sort_values("MSPB_Ratio")
    archetype_labels = ["High Value", "Cost Efficient / Lower Quality", "Average Performers",
                        "High Quality / Higher Cost", "Low Value", "Emerging Performers"]
    cluster_to_archetype = {}
    for rank, row in enumerate(centroids_sorted.itertuples()):
        cluster_to_archetype[row.cluster] = archetype_labels[rank] if rank < len(archetype_labels) else f"Cluster {rank}"

    label_series = pd.Series(labels).map(cluster_to_archetype)
    master["Cluster_Label"] = label_series.values

    # Print cluster profiles
    cluster_df["Cluster"] = labels
    cluster_df["Archetype"] = label_series.values
    print("\nCluster profiles:")
    print(cluster_df.groupby("Archetype")[CLUSTER_FEATURES].mean().round(3).to_string())
    print("\nCluster sizes:")
    print(cluster_df["Archetype"].value_counts().to_string())

    master.to_csv(PROCESSED / "master_hospital.csv", index=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kmeans": km, "cluster_to_archetype": cluster_to_archetype,
                 "cluster_features": CLUSTER_FEATURES}, MODELS / "kmeans_clusters.pkl")
    print("Saved models/kmeans_clusters.pkl and updated master_hospital.csv")
    return km, cluster_to_archetype


def explain_hospital(ccn: str) -> dict:
    data = joblib.load(MODELS / "xgb_payment_adj.pkl")
    reg = data["regressor"]
    feature_cols = data["feature_cols"]
    feat_data = joblib.load(PROCESSED / "features.pkl")
    X = feat_data["X"]

    master = pd.read_csv(PROCESSED / "master_hospital.csv", dtype=str)
    master["CCN"] = master["CCN"].str.zfill(6)
    ccn = str(ccn).zfill(6)
    idx = master[master["CCN"] == ccn].index

    if idx.empty:
        raise ValueError(f"Hospital CCN {ccn} not found")

    row = X.loc[idx[0]:idx[0], feature_cols].fillna(0)
    explainer = shap.TreeExplainer(reg)
    shap_values = explainer.shap_values(row)
    base_value = explainer.expected_value

    top_idx = np.argsort(np.abs(shap_values[0]))[::-1][:8]
    top_features = [(feature_cols[i], shap_values[0][i]) for i in top_idx]

    print(f"\nSHAP explanation for CCN {ccn}:")
    for feat, val in top_features:
        direction = "+" if val > 0 else ""
        print(f"  {feat}: {direction}{val:.4f}")

    return {"shap_values": shap_values, "base_value": base_value, "top_features": top_features}


if __name__ == "__main__":
    print("=== Training XGBoost Payment Predictor ===")
    train_payment_predictor()
    print("\n=== Training KMeans Cluster Model ===")
    train_cluster_model()
    print("\n=== SHAP Example (first hospital with known CCN) ===")
    master = pd.read_csv(PROCESSED / "master_hospital.csv", dtype=str)
    sample_ccn = master["CCN"].dropna().iloc[0]
    try:
        explain_hospital(sample_ccn)
    except Exception as e:
        print(f"SHAP example skipped: {e}")
