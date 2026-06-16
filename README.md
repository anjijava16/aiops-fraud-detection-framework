# aiops-fraud-detection-framework
1. An AIOps-driven scalable fraud detection framework for financial systems using XGBoost, LightGBM, CatBoost, SHAP Explainable AI, MLflow, FastAPI, concept drift detection, and automated model retraining.

# End to End Flow diagrms

Here's a comprehensive end-to-end architecture diagram for your `aiops-fraud-detection-framework` IEEE paper — covering all layers from data ingestion through AIOps monitoring:The diagram covers all 7 layers of your `aiops-fraud-detection-framework` end-to-end, color-coded by architectural concern:

**Layer 1 — Data ingestion (teal):** Kafka streaming for real-time transactions, batch CSV/DB for the Kaggle CCFD dataset, a feature store (V1–V28 PCA features), and a schema validator.

**Layer 2 — Preprocessing (purple):** SMOTE oversampling for class imbalance, normalization via StandardScaler/MinMaxScaler, correlation-based feature selection, and stratified 80/20 splitting with cross-validation.

**Layer 3 — Ensemble ML (coral):** XGBoost, LightGBM, and CatBoost as individual learners, unified through a soft-voting ensemble tuned for precision-recall balance on the imbalanced CCFD dataset.

**Layer 4 — Explainable AI (amber):** SHAP global (summary/bar plots), SHAP local (waterfall/force plots per prediction), LIME local surrogates, and a decision audit log for regulatory traceability.

**Layer 5 — MLflow (blue):** Experiment run tracking, staged model promotion in the registry, metric comparison (F1, AUC-ROC), and OpenTelemetry tracing via the Monocle SDK for observability.

**Layer 6 — FastAPI serving (green):** Three endpoints — `/predict` (fraud score 0–1), `/explain` (SHAP values in JSON), `/health` (liveness/readiness) — plus API key auth and rate limiting.

**Layer 7 — AIOps monitoring (pink):** PSI/KS-test drift detection, threshold-based alert pipeline (Slack/PagerDuty), automated retraining triggered by drift events, and a Grafana/Prometheus dashboard tracking live F1 and latency.

The dashed arrow on the left shows the **automated retrain feedback loop** — when drift is detected in Layer 7, new data flows back into Layer 1 and the pipeline re-executes through to model promotion. This loop is the core AIOps contribution you can highlight in your IEEE paper's contribution section.
