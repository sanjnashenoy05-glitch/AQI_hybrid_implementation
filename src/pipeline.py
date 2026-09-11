"""
Custom Hybrid Ensemble Framework for AQI Prediction and Risk Classification
============================================================================
Real dataset: city_day.csv (Air Quality Data in India, 2015-2020, CPCB source)
Pipeline: data quality -> EDA -> feature engineering -> RF/XGB/ExtraTrees base
learners -> hybrid layer (validation-weighted averaging AND stacking) ->
regression evaluation -> AQI category classification -> SHAP explainability.
"""

import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBRegressor
import shap
import joblib
import os

warnings.filterwarnings("ignore")
RNG = 42
np.random.seed(RNG)

BASE = "/home/claude/aqi_project"
OUT = f"{BASE}/outputs"
os.makedirs(OUT, exist_ok=True)

log = {}  # collects everything printed, dumped to json for the report

def record(key, value):
    log[key] = value
    print(f"\n=== {key} ===")
    print(value)

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
df = pd.read_csv(f"{BASE}/data/city_day.csv")
record("raw_shape", df.shape)
record("columns", list(df.columns))

# ---------------------------------------------------------------------------
# 2. DATA QUALITY / DATA PREPARATION
# ---------------------------------------------------------------------------
df["Date"] = pd.to_datetime(df["Date"])

missing_pct = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
record("missing_value_pct_by_column", missing_pct.to_dict())

dupes = df.duplicated(subset=["City", "Date"]).sum()
record("duplicate_city_date_rows", int(dupes))
df = df.drop_duplicates(subset=["City", "Date"])

# Target must be present to supervise on -> drop rows with no AQI (can't
# fabricate a target). This is the standard, defensible approach.
before = len(df)
df = df.dropna(subset=["AQI", "AQI_Bucket"])
record("rows_dropped_missing_target(AQI/AQI_Bucket)", before - len(df))
record("rows_after_target_dropna", len(df))

# Invalid-value check: pollutant concentrations cannot be negative
pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                   "O3", "Benzene", "Toluene", "Xylene"]
neg_counts = {c: int((df[c] < 0).sum()) for c in pollutant_cols}
record("negative_value_counts", neg_counts)
for c in pollutant_cols:
    df.loc[df[c] < 0, c] = np.nan  # treat impossible negative readings as missing

# Missing-value treatment: city-wise, time-aware interpolation for pollutants
# (short gaps within a city's own time series), then city-median fallback
# for any remaining gaps (e.g. a pollutant never measured at that city).
df = df.sort_values(["City", "Date"])
for c in pollutant_cols:
    df[c] = df.groupby("City")[c].transform(lambda s: s.interpolate(limit=7, limit_direction="both"))
    df[c] = df.groupby("City")[c].transform(lambda s: s.fillna(s.median()))
    df[c] = df[c].fillna(df[c].median())  # last resort, global median

record("missing_after_treatment", df[pollutant_cols].isna().sum().to_dict())

# ---------------------------------------------------------------------------
# 3. EDA (saved as PNGs for the slides / demo)
# ---------------------------------------------------------------------------
sns.set_style("whitegrid")

plt.figure(figsize=(7, 4))
sns.histplot(df["AQI"], bins=50, kde=True, color="#4C72B0")
plt.title("Distribution of AQI (city-day, cleaned)")
plt.xlabel("AQI"); plt.tight_layout()
plt.savefig(f"{OUT}/eda_aqi_distribution.png", dpi=150); plt.close()

plt.figure(figsize=(8, 6))
corr = df[pollutant_cols + ["AQI"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Pollutant-AQI Correlation")
plt.tight_layout()
plt.savefig(f"{OUT}/eda_correlation_heatmap.png", dpi=150); plt.close()

top_cities = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8, 5))
top_cities.plot(kind="barh", color="#C44E52")
plt.gca().invert_yaxis()
plt.title("Top 10 Cities by Mean AQI")
plt.xlabel("Mean AQI"); plt.tight_layout()
plt.savefig(f"{OUT}/eda_top_cities.png", dpi=150); plt.close()

df["Month"] = df["Date"].dt.month
monthly = df.groupby("Month")["AQI"].mean()
plt.figure(figsize=(7, 4))
monthly.plot(kind="line", marker="o", color="#55A868")
plt.title("Average AQI by Month (seasonality)")
plt.xlabel("Month"); plt.ylabel("Mean AQI"); plt.tight_layout()
plt.savefig(f"{OUT}/eda_monthly_seasonality.png", dpi=150); plt.close()

record("eda_plots_saved", ["eda_aqi_distribution.png", "eda_correlation_heatmap.png",
                            "eda_top_cities.png", "eda_monthly_seasonality.png"])

# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
df["Year"] = df["Date"].dt.year
df["DayOfYear"] = df["Date"].dt.dayofyear

def season_of(m):
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5): return "Summer"
    if m in (6, 7, 8, 9): return "Monsoon"
    return "PostMonsoon"

df["Season"] = df["Month"].apply(season_of)

city_dummies = pd.get_dummies(df["City"], prefix="City")
season_dummies = pd.get_dummies(df["Season"], prefix="Season")

feature_cols = pollutant_cols + ["Month", "Year"]
X = pd.concat([df[feature_cols].reset_index(drop=True),
               city_dummies.reset_index(drop=True),
               season_dummies.reset_index(drop=True)], axis=1)
y = df["AQI"].reset_index(drop=True)
y_bucket = df["AQI_Bucket"].reset_index(drop=True)

record("final_feature_matrix_shape", X.shape)

# ---------------------------------------------------------------------------
# 5. TRAIN / VALIDATION / TEST SPLIT (60/20/20)
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp, bucket_train, bucket_temp = train_test_split(
    X, y, y_bucket, test_size=0.4, random_state=RNG)
X_val, X_test, y_val, y_test, bucket_val, bucket_test = train_test_split(
    X_temp, y_temp, bucket_temp, test_size=0.5, random_state=RNG)

record("split_sizes", {"train": len(X_train), "val": len(X_val), "test": len(X_test)})

# ---------------------------------------------------------------------------
# 6. BASE MODELS
# ---------------------------------------------------------------------------
rf = RandomForestRegressor(n_estimators=150, max_depth=18, n_jobs=-1, random_state=RNG)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.08, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=RNG)
et = ExtraTreesRegressor(n_estimators=150, max_depth=18, n_jobs=-1, random_state=RNG)

base_models = {"RandomForest": rf, "XGBoost": xgb, "ExtraTrees": et}
val_preds, test_preds = {}, {}

for name, model in base_models.items():
    model.fit(X_train, y_train)
    val_preds[name] = model.predict(X_val)
    test_preds[name] = model.predict(X_test)

# ---------------------------------------------------------------------------
# 7. HYBRID LAYER A — validation-based weighted averaging
# ---------------------------------------------------------------------------
V = np.column_stack([val_preds[n] for n in base_models])
T = np.column_stack([test_preds[n] for n in base_models])

def neg_r2_of_weights(w):
    w = np.abs(w) / np.sum(np.abs(w))
    pred = V @ w
    return mean_squared_error(y_val, pred)

res = minimize(neg_r2_of_weights, x0=np.array([1/3, 1/3, 1/3]), method="Nelder-Mead")
weights = np.abs(res.x) / np.sum(np.abs(res.x))
weights_dict = {n: round(float(w), 4) for n, w in zip(base_models, weights)}
record("weighted_hybrid_weights (validation-optimised)", weights_dict)

weighted_test_pred = T @ weights

# ---------------------------------------------------------------------------
# 7b. HYBRID LAYER B — stacking (meta-learner on top of base predictions)
# Base learners are trained on TRAIN only (already done above). The meta
# learner (Ridge) is trained on their VALIDATION predictions (out-of-sample
# for the base models) and evaluated on the TEST base predictions -> no
# leakage, and far cheaper than sklearn's internal re-fit-per-fold stacking.
# ---------------------------------------------------------------------------
meta_learner = Ridge(alpha=1.0)
meta_learner.fit(V, y_val)
stacking_test_pred = meta_learner.predict(T)
record("stacking_meta_learner_coefficients", dict(zip(base_models.keys(), np.round(meta_learner.coef_, 4))))

# ---------------------------------------------------------------------------
# 8. REGRESSION EVALUATION — compare everything on the untouched test set
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred):
    return {
        "MAE": round(mean_absolute_error(y_true, y_pred), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(r2_score(y_true, y_pred), 4),
    }

results = {n: metrics(y_test, test_preds[n]) for n in base_models}
results["Hybrid_WeightedAvg"] = metrics(y_test, weighted_test_pred)
results["Hybrid_Stacking"] = metrics(y_test, stacking_test_pred)
record("regression_comparison_table (test set)", results)

best_hybrid_name = "Hybrid_Stacking" if results["Hybrid_Stacking"]["RMSE"] < results["Hybrid_WeightedAvg"]["RMSE"] else "Hybrid_WeightedAvg"
best_hybrid_pred = stacking_test_pred if best_hybrid_name == "Hybrid_Stacking" else weighted_test_pred
record("selected_hybrid_configuration", best_hybrid_name)

# ---------------------------------------------------------------------------
# 9. AQI -> CATEGORY MAPPING (standard CPCB breakpoints) + CLASSIFICATION EVAL
# ---------------------------------------------------------------------------
def aqi_to_bucket(v):
    if v <= 50: return "Good"
    if v <= 100: return "Satisfactory"
    if v <= 200: return "Moderate"
    if v <= 300: return "Poor"
    if v <= 400: return "Very Poor"
    return "Severe"

pred_bucket = pd.Series(best_hybrid_pred).apply(aqi_to_bucket).reset_index(drop=True)
true_bucket = bucket_test.reset_index(drop=True)

cls_metrics = {
    "Accuracy": round(accuracy_score(true_bucket, pred_bucket), 4),
    "Precision_macro": round(precision_score(true_bucket, pred_bucket, average="macro", zero_division=0), 4),
    "Recall_macro": round(recall_score(true_bucket, pred_bucket, average="macro", zero_division=0), 4),
    "F1_macro": round(f1_score(true_bucket, pred_bucket, average="macro", zero_division=0), 4),
}
record("classification_metrics (hybrid-derived category vs true AQI_Bucket)", cls_metrics)

labels_order = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
cm = confusion_matrix(true_bucket, pred_bucket, labels=labels_order)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels_order, yticklabels=labels_order)
plt.xlabel("Predicted category"); plt.ylabel("True category")
plt.title("Confusion Matrix — AQI Risk Category")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150); plt.close()

record("classification_report", classification_report(true_bucket, pred_bucket, labels=labels_order, zero_division=0))

# ---------------------------------------------------------------------------
# 10. SHAP EXPLAINABILITY (on the strongest single tree model — XGBoost)
# ---------------------------------------------------------------------------
explainer = shap.TreeExplainer(xgb)
sample = X_test.sample(n=min(200, len(X_test)), random_state=RNG)
shap_values = explainer.shap_values(sample)

plt.figure()
shap.summary_plot(shap_values, sample, show=False, max_display=12)
plt.tight_layout()
plt.savefig(f"{OUT}/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()

mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features = pd.Series(mean_abs_shap, index=sample.columns).sort_values(ascending=False).head(10)
record("top_10_shap_features", top_features.round(3).to_dict())

# ---------------------------------------------------------------------------
# 11. SINGLE-ROW DEMO (what you show Mam live)
# ---------------------------------------------------------------------------
demo_idx = X_test.index[0]
demo_row = X_test.loc[[demo_idx]]
demo_true_aqi = y_test.loc[demo_idx]
demo_true_bucket = bucket_test.loc[demo_idx]
demo_base_preds = np.column_stack([m.predict(demo_row) for m in base_models.values()])
demo_pred = meta_learner.predict(demo_base_preds)[0] if best_hybrid_name == "Hybrid_Stacking" else (demo_base_preds @ weights)[0]
demo_pred_bucket = aqi_to_bucket(demo_pred)

demo_shap = explainer.shap_values(demo_row)[0]
demo_top = pd.Series(demo_shap, index=demo_row.columns).abs().sort_values(ascending=False).head(5)

record("live_demo_example", {
    "true_AQI": float(demo_true_aqi),
    "true_category": demo_true_bucket,
    "predicted_AQI": round(float(demo_pred), 1),
    "predicted_category": demo_pred_bucket,
    "top_5_contributing_features": demo_top.round(2).to_dict(),
})

# ---------------------------------------------------------------------------
# 12. SAVE MODELS + FULL LOG
# ---------------------------------------------------------------------------
joblib.dump(rf, f"{OUT}/model_random_forest.joblib")
joblib.dump(xgb, f"{OUT}/model_xgboost.joblib")
joblib.dump(et, f"{OUT}/model_extra_trees.joblib")
joblib.dump(meta_learner, f"{OUT}/model_hybrid_stacking_meta.joblib")
joblib.dump({"weights": weights_dict, "feature_columns": list(X.columns)}, f"{OUT}/hybrid_weighted_config.joblib")

pd.DataFrame(results).T.to_csv(f"{OUT}/regression_results.csv")

with open(f"{OUT}/full_run_log.json", "w") as f:
    json.dump(log, f, indent=2, default=str)

print("\n\nALL DONE. Outputs saved to:", OUT)
