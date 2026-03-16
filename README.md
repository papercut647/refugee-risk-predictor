🌍 Refugee & Displacement Risk Predictor
An ML-powered dashboard for forecasting forced displacement driven by conflict, climate change, economic collapse, and governance breakdown.
Python 3.11+ • Streamlit • Scikit-Learn • MIT License

🎯 Overview
This interactive dashboard analyzes 18 indicators across 50 countries to classify displacement risk levels and forecast population movements through 2030. It combines machine learning with scenario-based projections grounded in verified UNHCR data.

⚠️ Disclaimer
This is a demonstration project built for portfolio purposes. It uses synthetic data modeled after real-world humanitarian datasets.

Forecasts are scenario-based projections, not predictions
Not validated for operational humanitarian planning
Built to demonstrate ML, data visualization, and domain modeling skills

For real displacement data, see UNHCR Global Trends

📊 Features
Risk Map

Global choropleth visualization of displacement risk scores
Country-level hover data with all indicator values
Risk class distribution (Critical / High / Medium / Low)
Top 15 countries by displaced persons

Forecasts

Scenario-based projections (2026–2030)
Multi-country comparison with interactive selection
UNHCR-grounded growth rates:

Conservative (3%): Peace progress, no new conflicts
Moderate (6%): Current crises persist (2024 baseline)
Pessimistic (8%): Escalation or new conflicts



Country Deep Dive

Individual country risk profiles
Radar chart across 5 risk domains
Time-series for displacement, risk score, conflict, and climate indicators
Scenario-responsive forecasts

Model Insights

Feature importance rankings
Correlation analysis with risk scores
Methodology documentation

Data Export

Filterable data table
One-click CSV download


🛠️ Technology Stack
ComponentTechnologyFrontendStreamlitData ProcessingPandas, NumPyMachine LearningScikit-Learn (Random Forest, Gradient Boosting)VisualizationPlotly Express & Graph ObjectsDeploymentStreamlit Community Cloud

📈 Methodology
Indicator Categories
CategoryWeightIndicatorsConflict30%Conflict intensity, ACLED events, battle deaths, territory lossClimate25%Temperature anomaly, flood risk, drought index, sea level rise, crop yield lossEconomic25%GDP per capita, food insecurity, Gini coefficient, unemployment, inflationGovernance20%Fragile States Index, press freedom, rule of law, corruption, political stability
Risk Classification
LevelScore RangeInterpretationCritical0.75 – 1.00Active conflict or imminent catastropheHigh0.55 – 0.74Multiple compounding risk factorsMedium0.35 – 0.54Elevated vulnerabilityLow0.00 – 0.34Relatively stable
Forecast Scenarios (UNHCR-Grounded)
Historical displacement growth rates inform the projections:
YearGrowthKey Driver2022+21%Ukraine crisis (largest ever)2023+8%Sudan civil war2024+6%123.2M displaced globally2025−1%Syrian returns (first decrease in decade)
Key Risk Factors Monitored:

Active Conflicts: Sudan, Myanmar, DRC, Sahel, Ukraine
Regional Tensions: Afghanistan-Pakistan, India-Pakistan (Kashmir), Ethiopia-Eritrea
Climate Shocks: Bangladesh flooding, Horn of Africa drought, Pacific sea-level rise
Governance Collapse: Haiti, Venezuela, Lebanon


🚀 Quick Start
Prerequisites

Python 3.11+
pip

Installation
bashgit clone https://github.com/papercut647/refugee-risk-predictor.git
cd refugee-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

### requirements.txt
```
streamlit
pandas
numpy
plotly
scikit-learn
```

---

## 📁 Project Structure
```
refugee-risk-predictor/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── docs/
    └── documentation.docx    # Detailed technical documentation

🔮 Future Roadmap
PriorityFeatureData SourceHighLive UNHCR API integrationUNHCR Data PortalHighACLED real-time conflict feedACLED APIHighWorld Bank WDI indicatorsWorld Bank APIMediumSub-national risk mappingUNOCHA HDXMediumSHAP explainability layerPython SHAPLowCustom scenario modelerUI controls

📚 Data Sources (Modeled After)
This dashboard uses synthetic data designed to mirror:

Climate: IPCC, EM-DAT, NASA GISS, CHIRPS
Conflict: ACLED, UCDP Armed Conflict Dataset
Economic: World Bank WDI, FAO, IMF
Governance: Fund for Peace (FSI), Transparency International, World Justice Project


📄 License
MIT License

Built to support evidence-based humanitarian planning and climate justice research.
