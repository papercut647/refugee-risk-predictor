# 🌍 Refugee & Displacement Risk Predictor

An ML-powered dashboard for forecasting forced displacement driven by conflict, climate change, economic collapse, and governance breakdown.

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Overview

This interactive dashboard analyzes **18 indicators across 50 countries** to classify displacement risk levels and forecast population movements through 2030. It combines machine learning with scenario-based projections grounded in verified UNHCR data.

---

## ⚠️ Disclaimer

This is a **demonstration project** built for portfolio purposes. It uses synthetic data modeled after real-world humanitarian datasets.

- Forecasts are scenario-based projections, not predictions
- Not validated for operational humanitarian planning
- Built to demonstrate ML, data visualization, and domain modeling skills

For real displacement data, see [UNHCR Global Trends](https://www.unhcr.org/global-trends)

---

## 📊 Features

### Risk Map
- Global choropleth visualization of displacement risk scores
- Country-level hover data with all indicator values
- Risk class distribution (Critical / High / Medium / Low)
- Top 15 countries by displaced persons

### Forecasts
- Scenario-based projections (2026–2030)
- Multi-country comparison with interactive selection
- UNHCR-grounded growth rates:
  - **Conservative (3%):** Peace progress, no new conflicts
  - **Moderate (6%):** Current crises persist (2024 baseline)
  - **Pessimistic (8%):** Escalation or new conflicts

### Country Deep Dive
- Individual country risk profiles
- Radar chart across 5 risk domains
- Time-series for displacement, risk score, conflict, and climate indicators
- Scenario-responsive forecasts

### Model Insights
- Feature importance rankings
- Correlation analysis with risk scores
- Methodology documentation

### Data Export
- Filterable data table
- One-click CSV download

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn (Random Forest, Gradient Boosting) |
| Visualization | Plotly Express & Graph Objects |
| Deployment | Streamlit Community Cloud |

---

## 📈 Methodology

### Indicator Categories

| Category | Weight | Indicators |
|----------|--------|------------|
| Conflict | 30% | Conflict intensity, ACLED events, battle deaths, territory loss |
| Climate | 25% | Temperature anomaly, flood risk, drought index, sea level rise, crop yield loss |
| Economic | 25% | GDP per capita, food insecurity, Gini coefficient, unemployment, inflation |
| Governance | 20% | Fragile States Index, press freedom, rule of law, corruption, political stability |

### Risk Classification

| Level | Score Range | Interpretation |
|-------|-------------|----------------|
| Critical | 0.75 – 1.00 | Active conflict or imminent catastrophe |
| High | 0.55 – 0.74 | Multiple compounding risk factors |
| Medium | 0.35 – 0.54 | Elevated vulnerability |
| Low | 0.00 – 0.34 | Relatively stable |

### Forecast Scenarios (UNHCR-Grounded)

Historical displacement growth rates inform the projections:

| Year | Growth | Key Driver |
|------|--------|------------|
| 2022 | +21% | Ukraine crisis (largest ever) |
| 2023 | +8% | Sudan civil war |
| 2024 | +6% | 123.2M displaced globally |
| 2025 | −1% | Syrian returns (first decrease in decade) |

**Key Risk Factors Monitored:**
- **Active Conflicts:** Sudan, Myanmar, DRC, Sahel, Ukraine
- **Regional Tensions:** Afghanistan-Pakistan, India-Pakistan (Kashmir), Ethiopia-Eritrea
- **Climate Shocks:** Bangladesh flooding, Horn of Africa drought, Pacific sea-level rise
- **Governance Collapse:** Haiti, Venezuela, Lebanon

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip

### Installation
```bash
git clone https://github.com/hammadmrza/refugee-risk-predictor.git
cd refugee-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure
```
refugee-risk-predictor/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔮 Future Roadmap

| Priority | Feature | Data Source |
|----------|---------|-------------|
| High | Live UNHCR API integration | UNHCR Data Portal |
| High | ACLED real-time conflict feed | ACLED API |
| High | World Bank WDI indicators | World Bank API |
| Medium | Sub-national risk mapping | UNOCHA HDX |
| Medium | SHAP explainability layer | Python SHAP |
| Low | Custom scenario modeler | UI controls |

---

## 📚 Data Sources (Modeled After)

This dashboard uses **synthetic data** designed to mirror:

- **Climate:** IPCC, EM-DAT, NASA GISS, CHIRPS
- **Conflict:** ACLED, UCDP Armed Conflict Dataset
- **Economic:** World Bank WDI, FAO, IMF
- **Governance:** Fund for Peace (FSI), Transparency International, World Justice Project

---

## 📄 License

MIT License

---

## 👤 Author

**Hammad Mirza**  
[LinkedIn](https://linkedin.com/in/hammadmirz) • [GitHub](https://github.com/hammadmrza)

---

*Built to support evidence-based humanitarian planning and climate justice research.*
