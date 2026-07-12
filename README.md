# 🏦 Bank Queue Wait-Time Prediction

Predicting customer waiting time in bank branches using simulation, feature engineering, and gradient-boosted ensemble models — with an interactive Streamlit dashboard for live predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![CatBoost](https://img.shields.io/badge/CatBoost-Gradient%20Boosting-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Long, unpredictable queues are one of the most common sources of customer dissatisfaction at bank branches. This project builds an end-to-end machine learning pipeline that predicts a customer's expected waiting time based on branch conditions at the moment of arrival — active counter count, current queue length, time of day, and requested service type.

The goal is not just a single model, but a **defensible ML workflow**: proper temporal validation, leakage checks, hyperparameter tuning, model interpretability (SHAP), and a usable interface for non-technical staff.

## 🎯 Key Features

- **Realistic simulation environment** (SimPy) to generate synthetic queue data for controlled experimentation, alongside a pipeline for real branch data (one month of operational records, ~11.8K visits).
- **Rich feature engineering**: cyclical time encoding (hour/day/minute as sin-cos pairs), peak-hour flags, queue-pressure indices, lag & rolling statistics, and train-only target encoding for service type.
- **Temporal train/test split** (not random) — the model is evaluated the way it would actually be used: trained on the past, tested on the future.
- **Model comparison**: CatBoost, XGBoost, LightGBM, Random Forest, and Gradient Boosting, combined into a **Stacking Ensemble** with a Ridge meta-learner.
- **Hyperparameter optimization** via Optuna (Bayesian search, 50 trials, 5-fold CV).
- **Interpretability**: SHAP summary, bar, and waterfall plots to explain both global feature importance and individual predictions.
- **Interactive dashboard** (Streamlit) for live "what-if" predictions, model evaluation charts, and data exploration — in Persian (RTL) for end users.

## 📊 Results

| Model | Test MAE (min) | R² | Overfit Gap |
|---|---|---|---|
| **Stacking Ensemble** | **2.18** | **0.869** | -0.02 |
| CatBoost (Optuna-tuned) | 2.19 | 0.869 | 0.06 |
| Random Forest | 2.21 | 0.865 | 0.46 |
| Gradient Boosting | 2.23 | 0.862 | 0.49 |
| LightGBM | 2.24 | 0.861 | 0.55 |
| XGBoost | 2.26 | 0.859 | 0.71 |

The Stacking Ensemble was selected as the final model: best test-set accuracy and near-zero train/test gap, indicating it generalizes well rather than memorizing the training period.

## 🧠 Methodology

1. **Data cleaning** — duplicate removal, missing-value handling, IQR-based outlier filtering on wait time.
2. **Feature engineering** — 59 engineered features across 7 groups (cyclical time, peak-hour flags, queue-load indices, log/sqrt transforms, interaction terms, lag features, rolling statistics).
3. **Leakage prevention**:
   - Target encoding for `Service_Type` computed **only on the training split**.
   - `RobustScaler` fit **only on the training split**.
   - Lag/rolling features use `shift(1)` to avoid look-ahead bias.
   - Any post-arrival column (service start/end time, actual service duration) is explicitly excluded from features.
4. **Temporal split** — first 80% of records (by date) for training, last 20% for testing, mimicking real deployment.
5. **Hyperparameter tuning** — Optuna minimizes 5-fold CV MAE for CatBoost.
6. **Ensembling** — a Stacking Regressor combines five base learners via a Ridge meta-model.
7. **Interpretability** — SHAP values quantify each feature's contribution, both in aggregate and per-prediction.

## 🗂️ Project Structure

```
.
├── notebooks/
│   └── Bank_Queue_WaitTime_Prediction.ipynb   # Full modeling pipeline (real data)
├── Wait_Time_Predictor_Enhanced.py             # Simulation + RandomForest baseline
├── streamlit_app_enhanced.py                   # Interactive prediction dashboard
├── bank_simulation_outputs/                    # Saved models & metrics (gitignored)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Installation

```bash
git clone https://github.com/sheyda2021/bank-queue-wait-time-prediction.git
cd bank-queue-wait-time-prediction
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook notebooks/Bank_Queue_WaitTime_Prediction.ipynb
```

### Run the dashboard

```bash
streamlit run streamlit_app_enhanced.py
```

## ⚠️ Known Limitations & Next Steps

Being upfront about limitations here rather than glossing over them:

- **Two datasets, two feature sets.** The simulation-based baseline (`Wait_Time_Predictor_Enhanced.py`) and the notebook trained on real one-month data use different feature schemas. The Streamlit dashboard currently serves the simpler simulation-based model; wiring up the notebook's Stacking Ensemble requires reconstructing lag/rolling/target-encoding features from historical context at inference time, not just swapping a pickle file. This is a planned next step, not yet implemented.
- **`Queue_Length` dominance.** In both pipelines, queue length at arrival (in its various transforms) is by far the strongest predictor — which makes sense physically (it's close to a direct proxy for wait time via basic queueing theory), but means the marginal value of the other engineered features is comparatively small. Reported for transparency rather than hidden behind a single R² number.
- **One month of real data.** The real-data model is trained and evaluated on a single month; seasonal effects (holidays, month-end banking rushes, etc.) are not captured and would need a longer observation window to validate.
- **No live A/B validation.** Model performance is measured on held-out historical data, not on live deployment; real-world drift (e.g., staffing changes, new services) isn't yet monitored.

## 🛠️ Tech Stack

`Python` · `pandas` / `NumPy` · `scikit-learn` · `CatBoost` · `XGBoost` · `LightGBM` · `Optuna` · `SHAP` · `SimPy` · `Streamlit` · `Matplotlib` / `Seaborn`


## 👤 Author

**Sheyda**
Graduate researcher working across ML, signal processing, and applied data science projects.
[GitHub](https://github.com/sheyda2021)

---

*If you find this project useful or have suggestions, feel free to open an issue or a pull request.*
