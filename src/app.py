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
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "data", "city_day.csv")

# ---------------------------------------------------------------------------
# Load pipeline log
# ---------------------------------------------------------------------------
if not os.path.exists(os.path.join(OUT, "full_run_log.json")):
    raise SystemExit(
        "outputs/ not found. Train the models first:\n"
        "    python src/pipeline.py\n"
        "then start the dashboard again:  python src/app.py"
    )

with open(os.path.join(OUT, "full_run_log.json")) as f:
    LOG = json.load(f)

# ---------------------------------------------------------------------------
# Load raw data for city medians + city list
# ---------------------------------------------------------------------------
RAW = pd.read_csv(DATA)
RAW["Date"] = pd.to_datetime(RAW["Date"])
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
              "O3", "Benzene", "Toluene", "Xylene"]
CITIES = sorted(RAW["City"].dropna().unique().tolist())
CITY_MEDIANS = RAW.groupby("City")[POLLUTANTS].median()
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

rf_clf = joblib.load(os.path.join(OUT, "model_random_forest_classifier.joblib"))
xgb_clf = joblib.load(os.path.join(OUT, "model_xgboost_classifier.joblib"))
et_clf = joblib.load(os.path.join(OUT, "model_extra_trees_classifier.joblib"))
meta_clf = joblib.load(os.path.join(OUT, "model_architectureB_stacking_meta.joblib"))
le = joblib.load(os.path.join(OUT, "label_encoder.joblib"))

base_models = {"RandomForest": rf, "XGBoost": xgb, "ExtraTrees": et}
clf_models = [rf_clf, xgb_clf, et_clf]

SELECTED_REG = weighted_cfg.get("selected_regression_hybrid", "Hybrid_Stacking")
SELECTED_CLS = weighted_cfg.get("selected_classification_hybrid", "DirectHybrid_Stacking")
cls_weights_cfg = weighted_cfg.get("classification_weights")
CLS_WEIGHTS = (np.array(list(cls_weights_cfg.values()), dtype=float)
               if cls_weights_cfg else np.array([1/3, 1/3, 1/3]))
CLS_WEIGHTS = CLS_WEIGHTS / CLS_WEIGHTS.sum()

# ---------------------------------------------------------------------------
# Feature construction — must match pipeline.py EXACTLY
# ---------------------------------------------------------------------------
def season_of(m):
    if m in (12, 1, 2): return "Winter"
    if m in (3, 4, 5): return "Summer"
    if m in (6, 7, 8, 9): return "Monsoon"
    return "PostMonsoon"

def build_feature_row(city, month, year):
    """Build a feature row from city + month + year only.
    Pollutant values are filled with the city's median (global median fallback)."""
    row = {}
    medians = CITY_MEDIANS.loc[city] if city in CITY_MEDIANS.index else GLOBAL_MEDIANS
    for p in POLLUTANTS:
        v = medians.get(p, GLOBAL_MEDIANS[p])
        if pd.isna(v):
            v = GLOBAL_MEDIANS[p]
        row[p] = float(v)
    row["Month"] = int(month)
    row["Year"] = int(year)

    for c in CITIES:
        row[f"City_{c}"] = 1 if c == city else 0

    season = season_of(int(month))
    for s in ["Winter", "Summer", "Monsoon", "PostMonsoon"]:
        row[f"Season_{s}"] = 1 if s == season else 0

    df_row = pd.DataFrame([row])
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
# SHAP explainer
# ---------------------------------------------------------------------------
import shap
SHAP_EXPLAINER = shap.TreeExplainer(xgb)

# ---------------------------------------------------------------------------
# Tab 1 — Prediction
# ---------------------------------------------------------------------------
def predict(city, month, year):
    X_row = build_feature_row(city, month, year)

    # --- Architecture A: regression -> threshold ---
    base_preds = np.array([[m.predict(X_row)[0] for m in base_models.values()]])
    if SELECTED_REG == "Hybrid_Stacking":
        aqi_pred = float(meta_reg.predict(base_preds)[0])
    else:
        aqi_pred = float((base_preds @ weights)[0])
    aqi_pred = max(0.0, aqi_pred)
    aqi_cat = aqi_to_bucket(aqi_pred)

    # --- Architecture B: direct classifier hybrid ---
    probas = [m.predict_proba(X_row) for m in clf_models]
    if SELECTED_CLS == "DirectHybrid_Stacking":
        stacked_input = np.hstack(probas)
        b_pred_enc = meta_clf.predict(stacked_input)[0]
        b_proba = meta_clf.predict_proba(stacked_input)[0]
        b_confidence = float(b_proba[b_pred_enc])
    else:
        blended = sum(w * p for w, p in zip(CLS_WEIGHTS, probas))
        b_pred_enc = int(blended.argmax(axis=1)[0])
        b_confidence = float(blended[0, b_pred_enc])
    b_cat = le.inverse_transform([b_pred_enc])[0]

    # --- SHAP for the demo row ---
    sv = SHAP_EXPLAINER.shap_values(X_row)[0]
    contrib = pd.Series(sv, index=X_row.columns).abs().sort_values(ascending=False).head(8)
    contrib = contrib[contrib > 0.01]

    # Agreement banner
    agree = (aqi_cat == b_cat)
    agree_html = (
        '<div style="margin:10px 0; padding:8px 12px; border-radius:6px; '
        'background:#d4edda; color:#155724; display:inline-block;">'
        '✓ Both architectures agree</div>'
        if agree else
        '<div style="margin:10px 0; padding:8px 12px; border-radius:6px; '
        'background:#fff3cd; color:#856404; display:inline-block;">'
        '⚠ Architectures disagree — A says <b>{}</b>, B says <b>{}</b></div>'.format(aqi_cat, b_cat)
    )

    # Year extrapolation notice
    year_notice = ""
    if int(year) > 2020:
        year_notice = (
            '<div style="margin:10px 0; padding:8px 12px; border-radius:6px; '
            'background:#e7f3ff; color:#004085; display:inline-block; font-size:13px;">'
            'ℹ Year {} is beyond the training range (2015–2020); tree models '
            'clamp to the 2020 boundary until retrained on newer data.</div>'.format(int(year))
        )

    html = f"""
    <div style="font-family:sans-serif; padding:10px;">
      <div style="display:flex; gap:20px; align-items:center; margin-bottom:10px; flex-wrap:wrap;">
        <div style="padding:20px; border-radius:12px; background:{BUCKET_COLORS[aqi_cat]}; color:white; min-width:220px;">
          <div style="font-size:14px; opacity:0.9;">Architecture A — regression → threshold</div>
          <div style="font-size:38px; font-weight:bold; line-height:1.1;">{aqi_pred:.0f}</div>
          <div style="font-size:22px; font-weight:600;">{aqi_cat}</div>
        </div>
        <div style="padding:20px; border-radius:12px; background:{BUCKET_COLORS[b_cat]}; color:white; min-width:220px;">
          <div style="font-size:14px; opacity:0.9;">Architecture B — direct classifier hybrid</div>
          <div style="font-size:38px; font-weight:bold; line-height:1.1;">{b_confidence:.0%}</div>
          <div style="font-size:22px; font-weight:600;">{b_cat}</div>
          <div style="font-size:12px; opacity:0.9;">classifier confidence</div>
        </div>
      </div>
      {agree_html}
      {year_notice}
      <h3 style="margin-top:20px;">Top contributing features (SHAP)</h3>
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
    a_wins = [m.replace("_macro", "") for m in ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"] if a[m] > b[m]]
    b_wins = [m.replace("_macro", "") for m in ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"] if b[m] > a[m]]
    if b_wins and not a_wins:
        verdict = f"Architecture B wins on {', '.join(b_wins)}."
    elif a_wins and not b_wins:
        verdict = f"Architecture A wins on {', '.join(a_wins)}."
    elif a_wins and b_wins:
        verdict = (f"Mixed result — Architecture B wins on {', '.join(b_wins)}; "
                   f"Architecture A wins on {', '.join(a_wins)}.")
    else:
        verdict = "The two architectures tie on every metric."
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
      <b>Verdict:</b> {verdict}
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
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])
_blocks_kwargs = {"title": "AQI Hybrid Ensemble — Demo"}
if _GRADIO_MAJOR < 6:
    _blocks_kwargs["theme"] = gr.themes.Soft()

with gr.Blocks(**_blocks_kwargs) as demo:
    gr.Markdown("# 🌫️ AQI Hybrid Ensemble — Interactive Demo")
    gr.Markdown("Custom hybrid framework for AQI prediction & risk classification (CPCB city_day dataset).")

    with gr.Tab("Predict"):
        gr.Markdown("Select a city and target month/year. Pollutant values are "
                    "automatically filled with that city's historical median.")
        with gr.Row():
            city = gr.Dropdown(choices=CITIES, value=CITIES[0], label="City")
            month = gr.Slider(1, 12, value=6, step=1, label="Month")
            year = gr.Slider(2026, 2031, value=2026, step=1, label="Year")
        btn = gr.Button("Predict", variant="primary")
        out = gr.HTML()
        btn.click(predict, inputs=[city, month, year], outputs=out)

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
    _launch_kwargs = {"server_name": "127.0.0.1", "server_port": 7860, "show_error": True}
    if _GRADIO_MAJOR >= 6:
        _launch_kwargs["theme"] = gr.themes.Soft()
    demo.launch(**_launch_kwargs)