🌍  Refugee & Displacement Risk Predictor
An ML-Powered Dashboard for Forecasting War & Climate Displacement
Version 1.0  •  March 2026  •  Built with Python, Streamlit & Scikit-Learn

🎯 What is this tool?
This dashboard uses machine learning to analyze 18 indicators across 50 countries and forecast which populations are at risk of forced displacement — whether driven by armed conflict, climate change, economic collapse, or governance breakdown. It covers historical data from 2010 to 2025, with forward projections through 2030.

1. Project Overview
Forced displacement is one of the most complex and urgent humanitarian challenges of the 21st century. At the end of 2023, the United Nations High Commissioner for Refugees (UNHCR) estimated that over 117 million people worldwide were forcibly displaced — the highest number ever recorded. These movements are driven by an increasingly intertwined combination of armed conflict, climate-related disasters, economic collapse, and government failure.

This dashboard was built to give researchers, policymakers, NGOs, and students a single, interactive tool to:
•	Visualize displacement risk across 50 countries on an interactive world map
•	Understand the relative contribution of climate, conflict, economic, and governance factors
•	Forecast displacement volumes and risk classifications out to 2030
•	Deep-dive into individual country profiles with time-series and radar visualizations
•	Explore the underlying machine learning models and their feature importance

2. Data Sources & Indicators
The dashboard is built on a representative synthetic dataset modeled after four major real-world data families. In a production deployment, each indicator would be replaced by direct API or download feeds from the sources listed below.

2.1  Climate Indicators
Climate data represents one of the fastest-growing drivers of displacement globally, particularly through the compounding effects of extreme weather events, sea-level rise, and agricultural disruption.

Indicator	Real-World Source	Description
Temperature Anomaly (°C)	IPCC / NASA GISS	Deviation from pre-industrial baseline; rising anomalies increase heat stress and drought
Flood Risk Index	EM-DAT / Global Flood Database	Probability of major flood events based on historical disaster records and hydrological modeling
Drought Index	CHIRPS / PDSI	Standardized Precipitation-Evapotranspiration Index reflecting soil moisture and water stress
Sea Level Rise (mm)	NOAA / Copernicus Marine Service	Cumulative sea-level change relevant to coastal displacement, particularly in delta regions
Crop Yield Loss (%)	FAO / FAOSTAT	Percentage decline in staple crop yields due to climate stress; strongly correlated with food insecurity

2.2  Conflict Indicators
Armed conflict remains the primary driver of displacement globally. The dashboard draws on two of the most authoritative real-time conflict tracking datasets in the world.

Indicator	Real-World Source	Description
Conflict Intensity Score	UCDP Armed Conflict Dataset	Composite measure of conflict severity, frequency, and geographic spread (0–1 scale)
ACLED Events Count	Armed Conflict Location & Event Data	Total number of political violence events including battles, explosions, and civilian targeting
Battle Deaths	UCDP Georeferenced Event Dataset	Estimated fatalities from organized political violence in a given year
Government Territory Loss	ACLED / ISW Analysis	Proportion of national territory under non-state or contested control

2.3  Economic Indicators
Economic fragility amplifies all other displacement drivers. Populations with few economic alternatives are far more likely to flee when conflict or climate shocks occur.

Indicator	Real-World Source	Description
GDP per Capita (USD)	World Bank WDI	Annual income per person; low GDP sharply reduces resilience to shocks
Food Insecurity Index	FAO SOFI Report	Share of population experiencing severe food insecurity (IPC Phase 3+)
Gini Coefficient	World Bank / UNDP HDR	Income inequality measure (0–100); higher scores indicate greater societal tension
Unemployment Rate	ILO / World Bank	Broad unemployment including informal sector; high rates fuel instability
Inflation Rate (%)	IMF World Economic Outlook	Annual consumer price inflation; hyperinflation is a key displacement trigger

2.4  Governance Indicators
Governance quality determines state capacity to protect populations, manage crises, and maintain territorial control. Weak governance multiplies the impact of every other risk factor.

Indicator	Real-World Source	Description
Fragile States Index	Fund for Peace (FFP)	Annual composite score (0–120) across 12 cohesion, economic, political, and social indicators
Press Freedom Index	Reporters Without Borders (RSF)	Inverse score of media repression; low press freedom signals authoritarian deterioration
Rule of Law Score	World Justice Project / World Bank WGI	Extent to which citizens and institutions abide by the law and courts function independently
Corruption Perceptions Index	Transparency International	Perceived public sector corruption (inverted for risk scoring)
Political Stability Score	World Bank Worldwide Governance Indicators	Likelihood of government destabilization or unconstitutional change (−2.5 to +2.5)

3. Machine Learning Models
3.1  Architecture Overview
The dashboard uses two separate machine learning models that work in tandem: one to classify a country's risk level, and one to estimate the volume of displaced persons.

Model	Algorithm	Target	Training Period
Risk Classifier	Random Forest (200 trees, max depth 12)	4-class: Critical / High / Medium / Low	2010–2023
Displacement Forecaster	Gradient Boosting Regressor (200 trees, lr=0.08)	Displaced persons count	2010–2023

3.2  Feature Engineering
All 18 features are standardized using a StandardScaler before being passed to either model. This ensures that features with different units (e.g., GDP in thousands vs. flood risk as a 0–1 score) contribute equally to model training. Features are grouped into four weighted categories:

Driver Category	Weight in Risk Score	Features Included
Conflict	30%	Conflict intensity, ACLED events, battle deaths, territory loss
Climate	25%	Temperature anomaly, flood risk, drought index, sea level rise, crop yield loss
Economic	25%	GDP per capita, food insecurity, Gini coefficient, unemployment, inflation
Governance	20%	Fragile States Index, press freedom, rule of law, corruption, political stability

3.3  Risk Classification Logic
The Random Forest classifier assigns one of four risk labels based on the composite risk score:

Risk Level	Risk Score Range	Interpretation
Critical	0.75 – 1.00	Active conflict or imminent climate catastrophe; mass displacement likely or underway
High	0.55 – 0.74	Multiple compounding risk factors; significant displacement expected within 1–2 years
Medium	0.35 – 0.54	Elevated vulnerability; displacement possible under additional shocks
Low	0.00 – 0.34	Relatively stable; displacement driven primarily by economic migration rather than forced flight

3.4  Forecast Methodology (2026–2030)
For the 5-year forecast, the 2025 indicator baseline for each country is projected forward using domain-informed trends:
•	Temperature anomaly: +0.08°C per year (IPCC RCP4.5 mid-range trajectory)
•	Flood risk, drought index, sea level rise: incremental annual increases based on current trend slopes
•	Food insecurity: gradually worsens in line with climate stress projections
•	Conflict intensity: held near the 2025 baseline with minor random variation (conflicts are harder to extrapolate)
•	GDP per capita: depreciates annually in proportion to conflict intensity (high-conflict countries face steeper declines)

The trained classifier and regressor are then applied to these projected feature sets to generate country-level risk classifications and displacement volume forecasts for each year through 2030.

3.5  Model Validation
Metric	Value	Notes
5-Fold Cross-Validation Accuracy	~85–92%	Evaluated on 2010–2023 training data with stratified folds
Countries in training set	50	Spanning all major displacement-affected regions
Total training records	700	50 countries × 14 years (2010–2023)
Held-out test period	2024–2025	Used only for visual validation in dashboard

4. Dashboard Features & Navigation
4.1  Sidebar Controls
The left sidebar provides global filters that affect all tabs simultaneously:
•	Year Slider — Scrub from 2010 to 2030; years after 2025 show forecast data
•	Region Filter — Multi-select to focus on specific geographic areas
•	Risk Level Filter — Filter to show only Critical, High, Medium, or Low risk countries
•	Forecast Toggle — Enable/disable the 2026–2030 projection period
•	Model Accuracy — Live display of cross-validation accuracy and standard deviation

4.2  Tab 1 — Risk Map
The Risk Map tab is the primary global overview. It contains three visualizations:
•	Global Choropleth Map — Countries shaded from green (low risk) to dark red (critical). Hover over any country to see all indicator values.
•	Countries by Risk Class (Donut Chart) — Shows the proportion of currently displayed countries in each risk tier.
•	Total Displaced by Risk Class (Bar Chart) — Translates risk classification into humanitarian scale.
•	Top 15 Countries by Displaced Persons — Horizontal bar chart with color-coded risk levels.

4.3  Tab 2 — Forecasts
The Forecasts tab provides time-series projections:
•	Multi-country line chart — Compare displacement trajectories for up to any combination of the 50 countries, with a shaded forecast zone after 2025.
•	Global stacked area chart — Shows total worldwide displacement broken down by risk class over time.
•	Climate vs. Conflict driver split — Distinguishes displacement attributable primarily to climate factors from those driven by conflict.

4.4  Tab 3 — Country Deep Dive
The Country Deep Dive tab provides a full analytical profile for any selected country:
•	Risk badge and key metrics — Instant summary of risk level, score, displaced persons, conflict intensity, flood risk, and food insecurity.
•	Risk Radar Chart — Pentagonal radar showing relative severity across all five driver domains simultaneously.
•	Displacement and Risk Score time series — Historical trend plus 2030 forecast with shaded uncertainty zone.
•	Conflict and Climate indicator panels — Detailed multi-line charts for each indicator category.

4.5  Tab 4 — Model Insights
This tab exposes the inner workings of the machine learning models:
•	Feature Importance (Random Forest) — Ranked bar chart showing which of the 18 indicators most influence the risk classification.
•	Feature Correlation with Risk Score — Pearson correlation chart showing positive and negative relationships.
•	Methodology table — Summary of all modeling decisions, data normalization, and forecast assumptions.

4.6  Tab 5 — Full Data Table
A filterable, sortable table of all country data for the selected year, with a one-click CSV export button for downstream analysis in Excel or R.

5. Deployment & Access
5.1  Running Locally
Requirements: Python 3.11+, pip

pip install streamlit pandas numpy scikit-learn plotly
 streamlit run refugee_predictor.py

The app will open automatically at http://localhost:8501. To view on a phone connected to the same Wi-Fi network, visit http://[your-local-IP]:8501.

5.2  Cloud Deployment (Streamlit Community Cloud)
•	Push refugee_predictor.py and requirements.txt to a public GitHub repository
•	Sign in at share.streamlit.io with your GitHub account
•	Select your repository, set the main file as refugee_predictor.py, and click Deploy
•	A permanent public URL is generated (e.g. https://username-refugee-risk-predictor.streamlit.app)
•	Free tier supports unlimited public apps with no time limit

6. Limitations & Ethical Considerations

⚠️  Important Disclaimer
This dashboard uses synthetic representative data modeled after real-world sources. It is intended for research, education, and humanitarian planning purposes only. Predictions should not be used as the sole basis for policy decisions affecting displaced populations without validation against current primary data.

Key limitations to be aware of:
•	Synthetic data: All indicator values are generated algorithmically to reflect known patterns and correlations. Real-world deployment requires connection to live data APIs.
•	Conflict forecasting uncertainty: Armed conflicts are inherently unpredictable. The 2026–2030 conflict projections should be treated as scenario baselines, not firm predictions.
•	Country selection: The 50-country sample covers the majority of global displacement but excludes some secondary displacement contexts.
•	Climate tipping points: The climate projections use linear trend extrapolation and do not account for non-linear tipping point dynamics.
•	Internal displacement: The model estimates total displaced persons but does not distinguish between internally displaced persons (IDPs) and cross-border refugees.

7. Future Development Roadmap

Priority	Feature	Data Source Required
High	Live UNHCR API integration for real displacement counts	UNHCR Data Portal API
High	ACLED real-time conflict event feed	ACLED API (free academic access)
High	World Bank WDI live indicators	World Bank Data API
Medium	Sub-national / subnational risk mapping	UNOCHA HDX, EM-DAT
Medium	SHAP explainability layer for individual country predictions	Python SHAP library
Medium	IDP vs. cross-border refugee split	UNHCR Population Statistics
Low	Scenario modeler (what if conflict intensifies?)	Custom parameter override UI
Low	Email/SMS alert system for risk threshold breaches	SendGrid / Twilio API
Low	PDF report auto-generation per country	ReportLab / WeasyPrint

8. Technology Stack

Component	Technology	Purpose
Frontend / UI	Streamlit 1.x	Interactive web dashboard, no JavaScript required
Data Processing	Pandas, NumPy	Data manipulation, feature engineering, aggregation
Machine Learning	Scikit-Learn	Random Forest classifier and Gradient Boosting regressor
Visualization	Plotly Express & Graph Objects	Choropleth maps, time series, radar charts, area plots
Deployment	Streamlit Community Cloud	Free, permanent public hosting via GitHub integration
Language	Python 3.11	Core runtime for all components

Built to support evidence-based humanitarian planning and climate justice research.
github.com/papercut647/refugee-risk-predictor
