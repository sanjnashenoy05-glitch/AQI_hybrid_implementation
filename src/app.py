"""
Gradio dashboard for the AQI Hybrid Ensemble project.
Run from the repo root:  python src/app.py
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data", "city_day.csv")

# ---------------------------------------------------------------------------
# Load pipeline log (has everything we need: cities, feature columns, metrics)
# ---------------------------------------------------------------------------
with open(os.path.join(OUT, "full_run_log.json")) as f:
    LOG = json.load(f)

# ---------------------------------------------------------------------------
# Load the raw data once for city medians + city list
# ---------------------------------------------------------------------------
RAW = pd.read_csv(DATA)
RAW["Date"] = pd.to_datetime(RAW["Date"])
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
              "O3", "Benzene", "Toluene", "Xylene"]
CITIES = sorted(RAW["City"].dropna().unique().tolist())
CITY_MEDIANS = RAW.groupby("City")[POLLUTANTS].median()

# Global medians as ultimate fallback
GLOBAL_MEDIANS = RAW[POLLUTANTS].median()

# ---------------------------------------------------------------------------
# Load trained models
# ---------------------------------------------------------------------------
rf = joblib.load(os.path.join(OUT, "model_random_forest.joblib"))
xgb = joblib.load(os.path.join(OUT, "model_xgboost.joblib"))
et = joblib.load(os.path.join(OUT, "model_extra_trees.joblib"))
meta_reg = joblib.load(os.path.join(OUT, "model_hybrid_stacking_meta.joblib"))
weighted_cfg = joblib.load(os.path.join(OUT, "hybrid_weighted_config.joblib"))
weights = np.array([weighted_cfg["weights"][n] for n in ["RandomForest", "XGBoost", "ExtraTrees"]])
feature_columns = weighted_cfg["feature_columns"]

xgb_clf = joblib.load(os.path.join(OUT, "model_xgboost_classifier.joblib"))
meta_clf = joblib.load(os.path.join(OUT, "model_architectureB_stacking_meta.joblib"))
le = joblib.load(os.path.join(OUT, "label_encoder.joblib"))

base_models = {"RandomForest": rf, "XGBoost": xgb, "ExtraTrees": et}

# ---------------------------------------------------------------------------
# Feature construction — must match pipeline.py EXACTLY
# ---------------------------------------------------------------------------
def season_of(m):
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5): return "Summer"
    if m in (6, 7, 8, 9): return "Monsoon"
    return "PostMonsoon"

def build_feature_row(city, month, year, pollutant_values):
    """pollutant_values is a dict of the 12 pollutants; missing -> city median."""
    row = {}
    medians = CITY_MEDIANS.loc[city] if city in CITY_MEDIANS.index else GLOBAL_MEDIANS
    for p in POLLUTANTS:
        v = pollutant_values.get(p)
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == 0:
            v = medians.get(p, GLOBAL_MEDIANS[p])
        row[p] = float(v)
    row["Month"] = int(month)
    row["Year"] = int(year)

    # city dummies
    for c in CITIES:
        row[f"City_{c}"] = 1 if c == city else 0

    # season dummies
    season = season_of(int(month))
    for s in ["Winter", "Summer", "Monsoon", "PostMonsoon"]:
        row[f"Season_{s}"] = 1 if s == season else 0

    df_row = pd.DataFrame([row])
    # align to the exact column order used in training
    df_row = df_row.reindex(columns=feature_columns, fill_value=0)
    return df_row

def aqi_to_bucket(v):
    if v <= 50:  return "Good"
    if v <= 100: return "Satisfactory"
    if v <= 200: return "Moderate"
    if v <= 300: return "Poor"
    if v <= 400: return "Very Poor"
    return "Severe"

BUCKET_COLORS = {
    "Good": "#00b050",
    "Satisfactory": "#92d050",
    "Moderate": "#ffc000",
    "Poor": "#ff6600",
    "Very Poor": "#c00000",
    "Severe": "#7f0000",
}

# ---------------------------------------------------------------------------
# SHAP explainer (built once at startup)
# ---------------------------------------------------------------------------
import shap
SHAP_EXPLAINER = shap.TreeExplainer(xgb)

# ---------------------------------------------------------------------------
# Tab 1 — Prediction
# ---------------------------------------------------------------------------
def predict(city, month, year, pm25, pm10, no, no2, nox, nh3, co, so2, o3,
            benzene, toluene, xylene):
    pv = {"PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NOx": nox,
          "NH3": nh3, "CO": co, "SO2": so2, "O3": o3,
          "Benzene": benzene, "Toluene": toluene, "Xylene": xylene}

    X_row = build_feature_row(city, month, year, pv)

    # --- Architecture A: regression -> threshold ---
    base_preds = np.column_stack([m.predict(X_row)[0] for m in base_models.values()])
    aqi_pred = float(meta_reg.predict(base_preds)[0])
    aqi_pred = max(0.0, aqi_pred)
    aqi_cat = aqi_to_bucket(aqi_pred)

    # --- Architecture B: direct classifier hybrid ---
    # Rebuild the classifier input the same way (same feature columns)
    cls_probas = []
    for name, model in [("RandomForest_clf", None), ("XGBoost_clf", xgb_clf), ("ExtraTrees_clf", None)]:
        pass
    # We only saved the XGBoost classifier + meta; simpler path: use xgb_clf only if the others aren't saved
    # But we DO have all three if pipeline saved them — check below.
    try:
        rf_clf = joblib.load(os.path.join(OUT, "model_random_forest_classifier.joblib"))
        et_clf = joblib.load(os.path.join(OUT, "model_extra_trees_classifier.joblib"))
        probas = np.hstack([
            rf_clf.predict_proba(X_row),
            xgb_clf.predict_proba(X_row),
            et_clf.predict_proba(X_row),
        ])
        b_pred_enc = meta_clf.predict(probas)[0]
        b_cat = le.inverse_transform([b_pred_enc])[0]
    except FileNotFoundError:
        # Fallback: use XGBoost classifier alone
        b_pred_enc = xgb_clf.predict(X_row)[0]
        b_cat = le.inverse_transform([b_pred_enc])[0]

    # --- SHAP for the demo row (using XGBoost regressor) ---
    sv = SHAP_EXPLAINER.shap_values(X_row)[0]
    contrib = pd.Series(sv, index=X_row.columns).abs().sort_values(ascending=False).head(8)
    contrib = contrib[contrib > 0.01]  # drop noise

    # Build output HTML
    html = f"""
    <div style="font-family:sans-serif; padding:10px;">
      <div style="display:flex; gap:20px; align-items:center; margin-bottom:20px;">
        <div style="padding:20px; border-radius:12px; background:{BUCKET_COLORS[aqi_cat]}; color:white; min-width:200px;">
          <div style="font-size:14px; opacity:0.9;">Architecture A (regression → threshold)</div>
          <div style="font-size:38px; font-weight:bold; line-height:1.1;">{aqi_pred:.0f}</div>
          <div style="font-size:22px; font-weight:600;">{aqi_cat}</div>
        </div>
        <div style="padding:20px; border-radius:12px; background:{BUCKET_COLORS[b_cat]}; color:white; min-width:200px;">
          <div style="font-size:14px; opacity:0.9;">Architecture B (direct classifier hybrid)</div>
          <div style="font-size:38px; font-weight:bold; line-height:1.1;">—</div>
          <div style="font-size:22px; font-weight:600;">{b_cat}</div>
        </div>
      </div>
      <h3>Top contributing features (SHAP)</h3>
      <table style="border-collapse:collapse;">
        <tr style="background:#eee;"><th style="padding:6px 12px; text-align:left;">Feature</th>
        <th style="padding:6px 12px; text-align:right;">|SHAP|</th></tr>
        {''.join(f'<tr><td style="padding:4px 12px;">{k}</td><td style="padding:4px 12px; text-align:right;">{v:.2f}</td></tr>' for k,v in contrib.items())}
      </table>
    </div>
    """
    return html

# ---------------------------------------------------------------------------
# Tab 2 — Architecture A vs B
# ---------------------------------------------------------------------------
def get_arch_comparison_html():
    a = LOG["ARCHITECTURE_A_vs_B_final_comparison (test set)"]["Architecture_A_metrics"]
    b = LOG["ARCHITECTURE_A_vs_B_final_comparison (test set)"]["Architecture_B_metrics"]
    rows = ""
    for metric in ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"]:
        av, bv = a[metric], b[metric]
        winner = "B" if bv > av else ("A" if av > bv else "tie")
        a_style = "background:#d4edda;font-weight:bold;" if winner == "A" else ""
        b_style = "background:#d4edda;font-weight:bold;" if winner == "B" else ""
        rows += f"""
        <tr>
          <td style="padding:8px 16px; border-bottom:1px solid #eee;">{metric.replace('_',' ')}</td>
          <td style="padding:8px 16px; border-bottom:1px solid #eee; {a_style}">{av:.4f}</td>
          <td style="padding:8px 16px; border-bottom:1px solid #eee; {b_style}">{bv:.4f}</td>
        </tr>"""
    return f"""
    <h2>Architecture A vs B — test set</h2>
    <p style="color:#555;">
      <b>A</b>: single hybrid regressor; category derived via fixed CPCB thresholds.<br>
      <b>B</b>: independent hybrid classifier trained directly on AQI_Bucket labels.
    </p>
    <table style="border-collapse:collapse; font-family:sans-serif;">
      <tr style="background:#333; color:white;">
        <th style="padding:8px 16px; text-align:left;">Metric</th>
        <th style="padding:8px 16px;">Architecture A</th>
        <th style="padding:8px 16px;">Architecture B</th>
      </tr>
      {rows}
    </table>
    <p style="margin-top:20px; font-size:15px;">
      <b>Verdict:</b> Architecture B wins on Accuracy, Recall, and F1 — deriving
      categories from a regressor loses information that a dedicated classifier
      can capture.
    </p>
    """

# ---------------------------------------------------------------------------
# Tab 3 — Model comparison
# ---------------------------------------------------------------------------
def get_model_comparison_html():
    res = LOG["regression_comparison_table (test set)"]
    rows = ""
    best_rmse = min(v["RMSE"] for v in res.values())
    for name, m in res.items():
        style = "background:#d4edda;font-weight:bold;" if m["RMSE"] == best_rmse else ""
        rows += f"""
        <tr>
          <td style="padding:6px 14px; border-bottom:1px solid #eee;">{name}</td>
          <td style="padding:6px 14px; border-bottom:1px solid #eee; text-align:right;">{m['MAE']:.3f}</td>
          <td style="padding:6px 14px; border-bottom:1px solid #eee; text-align:right; {style}">{m['RMSE']:.3f}</td>
          <td style="padding:6px 14px; border-bottom:1px solid #eee; text-align:right;">{m['R2']:.4f}</td>
        </tr>"""
    top_shap = LOG.get("top_10_shap_features", {})
    shap_rows = "".join(
        f"<tr><td style='padding:4px 12px;'>{k}</td><td style='padding:4px 12px; text-align:right;'>{v:.2f}</td></tr>"
        for k, v in list(top_shap.items())[:8]
    )
    return f"""
    <h2>Regression model comparison (test set)</h2>
    <table style="border-collapse:collapse; font-family:sans-serif;">
      <tr style="background:#333; color:white;">
        <th style="padding:6px 14px; text-align:left;">Model</th>
        <th style="padding:6px 14px;">MAE</th>
        <th style="padding:6px 14px;">RMSE</th>
        <th style="padding:6px 14px;">R²</th>
      </tr>
      {rows}
    </table>
    <h2 style="margin-top:30px;">Top SHAP features (XGBoost)</h2>
    <table style="border-collapse:collapse; font-family:sans-serif;">
      <tr style="background:#333; color:white;">
        <th style="padding:4px 12px; text-align:left;">Feature</th>
        <th style="padding:4px 12px;">Mean |SHAP|</th>
      </tr>
      {shap_rows}
    </table>
    """

# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AQI Hybrid Ensemble — Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌫️ AQI Hybrid Ensemble — Interactive Demo")
    gr.Markdown("Custom hybrid framework for AQI prediction & risk classification (CPCB city_day dataset).")

    with gr.Tab("Predict"):
        gr.Markdown("Fill in the city, date, and any pollutants you know. "
                    "Blanks fall back to that city's median.")
        with gr.Row():
            city = gr.Dropdown(choices=CITIES, value=CITIES[0], label="City")
            month = gr.Slider(1, 12, value=6, step=1, label="Month")
            year = gr.Slider(2015, 2020, value=2019, step=1, label="Year")
        with gr.Row():
            pm25 = gr.Number(label="PM2.5", value=None)
            pm10 = gr.Number(label="PM10", value=None)
            no   = gr.Number(label="NO", value=None)
            no2  = gr.Number(label="NO2", value=None)
        with gr.Row():
            nox  = gr.Number(label="NOx", value=None)
            nh3  = gr.Number(label="NH3", value=None)
            co   = gr.Number(label="CO", value=None)
            so2  = gr.Number(label="SO2", value=None)
        with gr.Row():
            o3       = gr.Number(label="O3", value=None)
            benzene  = gr.Number(label="Benzene", value=None)
            toluene  = gr.Number(label="Toluene", value=None)
            xylene   = gr.Number(label="Xylene", value=None)
        btn = gr.Button("Predict", variant="primary")
        out = gr.HTML()
        btn.click(predict,
                  inputs=[city, month, year, pm25, pm10, no, no2, nox, nh3,
                          co, so2, o3, benzene, toluene, xylene],
                  outputs=out)

    with gr.Tab("Architecture A vs B"):
        gr.HTML(get_arch_comparison_html())
        with gr.Row():
            gr.Image(os.path.join(OUT, "confusion_matrix.png"),
                     label="Architecture A — confusion matrix")
            gr.Image(os.path.join(OUT, "confusion_matrix_architecture_B.png"),
                     label="Architecture B — confusion matrix")

    with gr.Tab("Model comparison"):
        gr.HTML(get_model_comparison_html())
        gr.Image(os.path.join(OUT, "shap_summary.png"), label="SHAP summary (XGBoost)")

    with gr.Tab("EDA"):
        with gr.Row():
            gr.Image(os.path.join(OUT, "eda_aqi_distribution.png"), label="AQI distribution")
            gr.Image(os.path.join(OUT, "eda_correlation_heatmap.png"), label="Correlation heatmap")
        with gr.Row():
            gr.Image(os.path.join(OUT, "eda_top_cities.png"), label="Top cities by mean AQI")
            gr.Image(os.path.join(OUT, "eda_monthly_seasonality.png"), label="Seasonality")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)