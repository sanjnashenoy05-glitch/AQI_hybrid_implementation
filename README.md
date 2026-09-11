# Custom Hybrid Ensemble Framework for AQI Prediction and Risk Classification

Implementation only — run it yourself to get your own output (plots, metrics, models).

## Files
- `AQI_Hybrid_Prototype.ipynb` — run all cells top to bottom.
- `src/pipeline.py` — same pipeline as a script (`python3 src/pipeline.py`), saves plots/metrics/models to an `outputs/` folder it creates.
- `data/city_day.csv` — real Air Quality Data in India dataset (city-day level, CPCB source).
- `requirements.txt` — dependencies.

## Setup
```
pip install -r requirements.txt
python3 src/pipeline.py
```
or open the notebook and run all cells.

## What it does
Data quality checks -> cleaning (missing-value treatment, duplicate/invalid-value handling) ->
EDA -> feature engineering (pollutants + month/year + city + season) -> three base learners
(Random Forest, XGBoost, Extra Trees) -> hybrid layer (validation-weighted averaging AND
stacking, both computed so you can compare) -> regression evaluation (MAE/RMSE/R2) -> AQI ->
risk-category mapping and classification evaluation (accuracy/precision/recall/F1/confusion
matrix) -> SHAP explainability -> single-row live-demo prediction.
