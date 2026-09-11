# Custom Hybrid Ensemble Framework for AQI Prediction and Risk Classification

Implementation only — run it yourself to get your own output (plots, metrics, models).

## Two architectures, both implemented and compared
- **Architecture A**: one hybrid regressor (RF + XGBoost + Extra Trees) predicts numeric AQI;
  risk category is *derived* from that number via standard CPCB thresholds.
- **Architecture B**: a second, independent hybrid — RF/XGBoost/Extra Trees *classifiers* —
  trained directly on the true AQI_Bucket labels, combined via validation-weighted probability
  averaging and stacking (Logistic Regression meta-learner).
Both hybrid layers (weighted averaging AND stacking) are computed for regression AND for
classification, and Architecture A vs B are compared head-to-head on the same held-out test set.

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

## Pipeline
Data quality checks -> cleaning (missing-value treatment, duplicate/invalid-value handling) ->
EDA -> feature engineering (pollutants + month/year + city + season) -> three base learners
(Random Forest, XGBoost, Extra Trees) -> Architecture A: regression hybrid (validation-weighted
averaging AND stacking) -> AQI -> risk-category mapping -> Architecture B: independent
classification hybrid trained directly on category labels (weighted probability averaging AND
stacking) -> Architecture A vs B comparison -> SHAP explainability -> single-row live demo.
