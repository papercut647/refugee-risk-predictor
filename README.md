# 🌍 Refugee & Displacement Risk Predictor

An ML-powered interactive dashboard forecasting war and climate refugee displacement risk across 50 countries from 2010–2030.

## What It Does
- 🗺 **Risk Map** — Interactive world map showing displacement risk by country
- 📈 **Forecasts** — Displacement projections through 2030 with climate vs. conflict breakdown
- 🔍 **Country Deep Dive** — Radar charts, time series, and per-indicator breakdowns
- ⚙️ **Model Insights** — Feature importance and correlation analysis
- 📋 **Data Table** — Filterable dataset with CSV export

## How It Works
Two ML models run in tandem — a **Random Forest classifier** (risk level) and a **Gradient Boosting regressor** (displaced persons count) — trained on 18 indicators across 4 driver categories:

| Category | Weight | Indicators |
|---|---|---|
| 🔴 Conflict | 30% | Conflict intensity, ACLED events, battle deaths, territory loss |
| 🌡 Climate | 25% | Temp anomaly, flood risk, drought index, sea level rise, crop yield loss |
| 💰 Economic | 25% | GDP per capita, food insecurity, Gini coefficient, unemployment, inflation |
| 🏛 Governance | 20% | Fragile States Index, press freedom, rule of law, corruption, political stability |

## Run Locally
```bash
pip install streamlit pandas numpy scikit-learn plotly
streamlit run refugee_predictor.py
```

## Data Sources
Climate (IPCC, NASA, EM-DAT) · Conflict (ACLED, UCDP) · Economic (World Bank, FAO, IMF) · Governance (Fund for Peace, Transparency International)

> ⚠️ Built on synthetic representative data modeled after real-world sources. For research and humanitarian planning use only.
