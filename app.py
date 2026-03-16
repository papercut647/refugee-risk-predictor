"""
Refugee & Displacement Risk Predictor
Combines climate, conflict, economic, and governance indicators
to forecast displacement volumes and classify country risk levels.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Refugee & Displacement Risk Predictor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border: 1px solid #2e3350;
    }
    .risk-critical { color: #ff4b4b; font-weight: 700; }
    .risk-high     { color: #ff8c00; font-weight: 700; }
    .risk-medium   { color: #ffd700; font-weight: 700; }
    .risk-low      { color: #00cc88; font-weight: 700; }
    h1 { color: #e0e6ff; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FORECAST SCENARIOS
# ─────────────────────────────────────────────
SCENARIOS = {
    "Conservative (3% annual growth)": {
        "rate": 0.03,
        "description": "Assumes continued peace in Syria, diplomatic progress in Sudan, stable conditions in South Asia, and no major new conflicts. Reflects the slowdown observed in late 2024 and early 2025.",
        "color": "#00cc88"
    },
    "Moderate (6% annual growth)": {
        "rate": 0.06,
        "description": "Based on 2024's verified UNHCR growth rate. Assumes current crises persist at similar intensity — Sudan, DRC, Myanmar, Afghanistan instability. No major new conflicts or resolutions.",
        "color": "#ffd700"
    },
    "Pessimistic (8% annual growth)": {
        "rate": 0.08,
        "description": "Based on 2023's growth rate. Assumes escalation in existing hotspots or new conflict emergence — Afghanistan-Pakistan border tensions, India-Pakistan escalation, renewed Ethiopian conflict, or major climate disasters.",
        "color": "#ff4b4b"
    }
}


# ─────────────────────────────────────────────
# DATA GENERATION (realistic synthetic data)
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)

    countries = {
        # (region, base_conflict, base_climate, base_econ, base_gov)
        "Syria":            ("Middle East",    0.92, 0.65, 0.88, 0.85),
        "Afghanistan":      ("South Asia",     0.90, 0.70, 0.90, 0.88),
        "South Sudan":      ("Sub-Saharan Africa", 0.85, 0.80, 0.92, 0.87),
        "Somalia":          ("Sub-Saharan Africa", 0.83, 0.82, 0.91, 0.86),
        "DR Congo":         ("Sub-Saharan Africa", 0.78, 0.60, 0.86, 0.82),
        "Yemen":            ("Middle East",    0.88, 0.72, 0.89, 0.84),
        "Myanmar":          ("Southeast Asia", 0.75, 0.65, 0.70, 0.78),
        "Sudan":            ("Sub-Saharan Africa", 0.80, 0.78, 0.82, 0.83),
        "CAR":              ("Sub-Saharan Africa", 0.77, 0.58, 0.88, 0.84),
        "Ethiopia":         ("Sub-Saharan Africa", 0.72, 0.74, 0.75, 0.76),
        "Mozambique":       ("Sub-Saharan Africa", 0.52, 0.78, 0.70, 0.68),
        "Haiti":            ("Caribbean",      0.60, 0.75, 0.80, 0.75),
        "Venezuela":        ("Latin America",  0.48, 0.40, 0.82, 0.72),
        "Nigeria":          ("Sub-Saharan Africa", 0.65, 0.70, 0.65, 0.70),
        "Mali":             ("Sub-Saharan Africa", 0.68, 0.75, 0.72, 0.74),
        "Burkina Faso":     ("Sub-Saharan Africa", 0.70, 0.72, 0.71, 0.73),
        "Niger":            ("Sub-Saharan Africa", 0.62, 0.80, 0.78, 0.71),
        "Chad":             ("Sub-Saharan Africa", 0.65, 0.77, 0.76, 0.72),
        "Libya":            ("Middle East",    0.68, 0.45, 0.58, 0.76),
        "Iraq":             ("Middle East",    0.60, 0.68, 0.55, 0.66),
        "Ukraine":          ("Europe",         0.82, 0.30, 0.60, 0.48),
        "Bangladesh":       ("South Asia",     0.22, 0.85, 0.48, 0.42),
        "Pakistan":         ("South Asia",     0.55, 0.72, 0.58, 0.62),
        "Philippines":      ("Southeast Asia", 0.38, 0.82, 0.42, 0.40),
        "Cambodia":         ("Southeast Asia", 0.20, 0.65, 0.45, 0.52),
        "Zimbabwe":         ("Sub-Saharan Africa", 0.35, 0.62, 0.72, 0.68),
        "Honduras":         ("Latin America",  0.48, 0.65, 0.62, 0.60),
        "Guatemala":        ("Latin America",  0.44, 0.68, 0.60, 0.58),
        "El Salvador":      ("Latin America",  0.50, 0.62, 0.58, 0.56),
        "Colombia":         ("Latin America",  0.52, 0.45, 0.50, 0.48),
        "Indonesia":        ("Southeast Asia", 0.25, 0.70, 0.38, 0.35),
        "India":            ("South Asia",     0.30, 0.68, 0.35, 0.30),
        "Brazil":           ("Latin America",  0.28, 0.55, 0.42, 0.36),
        "Cameroon":         ("Sub-Saharan Africa", 0.58, 0.60, 0.65, 0.64),
        "Kenya":            ("Sub-Saharan Africa", 0.40, 0.65, 0.55, 0.52),
        "Senegal":          ("Sub-Saharan Africa", 0.28, 0.60, 0.50, 0.38),
        "Eritrea":          ("Sub-Saharan Africa", 0.45, 0.55, 0.70, 0.78),
        "Belarus":          ("Europe",         0.32, 0.20, 0.45, 0.65),
        "Cuba":             ("Caribbean",      0.18, 0.52, 0.62, 0.55),
        "Iran":             ("Middle East",    0.42, 0.55, 0.60, 0.62),
        "Turkey":           ("Middle East",    0.38, 0.48, 0.50, 0.45),
        "Tunisia":          ("Middle East",    0.28, 0.50, 0.48, 0.40),
        "Morocco":          ("Middle East",    0.18, 0.52, 0.42, 0.35),
        "Jordan":           ("Middle East",    0.20, 0.55, 0.45, 0.32),
        "Ghana":            ("Sub-Saharan Africa", 0.22, 0.55, 0.42, 0.28),
        "Tanzania":         ("Sub-Saharan Africa", 0.25, 0.60, 0.48, 0.40),
        "Uganda":           ("Sub-Saharan Africa", 0.35, 0.55, 0.58, 0.54),
        "Peru":             ("Latin America",  0.25, 0.48, 0.45, 0.42),
        "Mexico":           ("Latin America",  0.45, 0.52, 0.48, 0.50),
        "Lebanon":          ("Middle East",    0.55, 0.42, 0.72, 0.70),
    }

    years = list(range(2010, 2026))
    records = []

    for country, (region, c_base, cl_base, e_base, g_base) in countries.items():
        prev_displaced = np.random.randint(5000, 500000) * c_base
        for yr in years:
            t = (yr - 2010) / 15.0

            # ── Climate indicators ──
            temp_anomaly       = cl_base * 1.8 + t * 0.15 + np.random.normal(0, 0.12)
            flood_risk         = min(1, cl_base * 0.9 + t * 0.08 + np.random.normal(0, 0.06))
            drought_index      = min(1, cl_base * 0.85 + t * 0.06 + np.random.normal(0, 0.07))
            sea_level_rise_mm  = 3.2 * (yr - 2000) + cl_base * 15 + np.random.normal(0, 3)
            crop_yield_loss    = max(0, cl_base * 0.6 + t * 0.05 + np.random.normal(0, 0.05))

            # ── Conflict indicators ──
            conflict_intensity = min(1, c_base + np.random.normal(0, 0.06) + (0.05 * t if yr > 2019 else 0))
            acled_events       = int(conflict_intensity * 800 + np.random.normal(0, 40))
            battle_deaths      = int(conflict_intensity * 15000 * np.random.lognormal(0, 0.5))
            gov_territory_loss = min(1, max(0, c_base * 0.7 + np.random.normal(0, 0.08)))

            # ── Economic indicators ──
            gdp_per_capita     = max(200, 12000 * (1 - e_base) + np.random.normal(0, 400))
            food_insecurity    = min(1, e_base * 0.85 + np.random.normal(0, 0.05))
            gini_coeff         = 25 + e_base * 45 + np.random.normal(0, 2.5)
            unemployment       = min(1, max(0, e_base * 0.6 + np.random.normal(0, 0.05)))
            inflation_rate     = max(0, e_base * 40 + np.random.normal(0, 5))

            # ── Governance indicators ──
            fragile_state_idx  = min(120, g_base * 110 + np.random.normal(0, 4))
            press_freedom      = max(0, 1 - g_base + np.random.normal(0, 0.06))
            rule_of_law        = max(0, 1 - g_base + np.random.normal(0, 0.06))
            corruption_idx     = min(1, g_base * 0.9 + np.random.normal(0, 0.05))
            political_stability= max(-3, -g_base * 3 + np.random.normal(0, 0.3))

            # ── Composite risk score ──
            risk_score = (
                0.30 * conflict_intensity +
                0.25 * cl_base +
                0.25 * e_base +
                0.20 * g_base
            ) + np.random.normal(0, 0.03)
            risk_score = min(1, max(0, risk_score))

            # ── Displacement volume ──
            displaced = max(0,
                prev_displaced * 0.85 +
                conflict_intensity * 600000 +
                flood_risk * 120000 +
                food_insecurity * 80000 +
                np.random.normal(0, 15000)
            )
            prev_displaced = displaced

            # ── Risk class ──
            if risk_score >= 0.75:   risk_class = "Critical"
            elif risk_score >= 0.55: risk_class = "High"
            elif risk_score >= 0.35: risk_class = "Medium"
            else:                    risk_class = "Low"

            records.append({
                "country": country, "region": region, "year": yr,
                # Climate
                "temp_anomaly": round(temp_anomaly, 2),
                "flood_risk": round(flood_risk, 3),
                "drought_index": round(drought_index, 3),
                "sea_level_rise_mm": round(sea_level_rise_mm, 1),
                "crop_yield_loss": round(crop_yield_loss, 3),
                # Conflict
                "conflict_intensity": round(conflict_intensity, 3),
                "acled_events": acled_events,
                "battle_deaths": battle_deaths,
                "gov_territory_loss": round(gov_territory_loss, 3),
                # Economic
                "gdp_per_capita": round(gdp_per_capita, 0),
                "food_insecurity": round(food_insecurity, 3),
                "gini_coeff": round(gini_coeff, 1),
                "unemployment": round(unemployment, 3),
                "inflation_rate": round(inflation_rate, 1),
                # Governance
                "fragile_state_idx": round(fragile_state_idx, 1),
                "press_freedom": round(press_freedom, 3),
                "rule_of_law": round(rule_of_law, 3),
                "corruption_idx": round(corruption_idx, 3),
                "political_stability": round(political_stability, 2),
                # Targets
                "risk_score": round(risk_score, 4),
                "risk_class": risk_class,
                "displaced_persons": int(displaced),
            })

    return pd.DataFrame(records)


@st.cache_data
def train_models(df):
    features = [
        "temp_anomaly", "flood_risk", "drought_index", "crop_yield_loss",
        "conflict_intensity", "acled_events", "battle_deaths", "gov_territory_loss",
        "gdp_per_capita", "food_insecurity", "gini_coeff", "unemployment", "inflation_rate",
        "fragile_state_idx", "press_freedom", "rule_of_law", "corruption_idx", "political_stability"
    ]

    train = df[df["year"] <= 2023].copy()
    X = train[features]
    y_class = train["risk_class"]
    y_reg   = train["displaced_persons"]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_s, y_class)

    reg = GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.08, random_state=42)
    reg.fit(X_s, y_reg)

    cv_scores = cross_val_score(clf, X_s, y_class, cv=5, scoring="accuracy")

    importances = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)

    return clf, reg, scaler, features, cv_scores, importances


@st.cache_data
def forecast_future(df, _clf, _reg, _scaler, features, growth_rate=0.06):
    """Project displacement forward to 2030 using scenario-based growth rates."""
    future_years = [2026, 2027, 2028, 2029, 2030]
    base = df[df["year"] == 2025].copy()
    
    # Map growth rate to conflict trend factor
    # Higher displacement growth = more conflict escalation
    if growth_rate <= 0.03:
        conflict_trend = -0.01  # Slight improvement
    elif growth_rate <= 0.06:
        conflict_trend = 0.005  # Slight worsening
    else:
        conflict_trend = 0.015  # Escalation

    rows = []
    for _, row in base.iterrows():
        base_displaced = row["displaced_persons"]
        base_conflict = row["conflict_intensity"]
        base_acled = row["acled_events"]
        base_deaths = row["battle_deaths"]
        base_territory = row["gov_territory_loss"]
        
        for i, yr in enumerate(future_years, 1):
            proj = row.copy()
            proj["year"] = yr
            
            # ── Climate indicators (always trending worse) ──
            proj["temp_anomaly"]       += i * 0.08
            proj["flood_risk"]          = min(1, proj["flood_risk"] + i * 0.015)
            proj["drought_index"]       = min(1, proj["drought_index"] + i * 0.012)
            proj["sea_level_rise_mm"]  += i * 3.5
            proj["crop_yield_loss"]     = min(1, proj["crop_yield_loss"] + i * 0.010)
            
            # ── Conflict indicators (scenario-dependent) ──
            proj["conflict_intensity"]  = min(1, max(0, base_conflict + i * conflict_trend + np.random.normal(0, 0.02)))
            proj["acled_events"]        = max(0, int(base_acled * (1 + conflict_trend * i) + np.random.normal(0, 20)))
            proj["battle_deaths"]       = max(0, int(base_deaths * (1 + conflict_trend * i * 0.5) + np.random.normal(0, 500)))
            proj["gov_territory_loss"]  = min(1, max(0, base_territory + i * conflict_trend * 0.5))
            
            # ── Economic indicators ──
            proj["gdp_per_capita"]     *= (1 - proj["conflict_intensity"] * 0.02)
            proj["food_insecurity"]     = min(1, proj["food_insecurity"] + i * 0.008)
            proj["unemployment"]        = min(1, proj["unemployment"] + i * 0.005 * (1 + conflict_trend))
            proj["inflation_rate"]      = max(0, proj["inflation_rate"] + i * 2 * proj["conflict_intensity"])
            
            # ── Governance indicators ──
            proj["fragile_state_idx"]   = min(120, proj["fragile_state_idx"] + i * 0.5 * (1 + conflict_trend * 10))
            proj["political_stability"] = max(-3, proj["political_stability"] - i * 0.05 * (1 + conflict_trend * 5))
            proj["corruption_idx"]      = min(1, proj["corruption_idx"] + i * 0.005)
            
            # ── Calculate risk score (weighted composite) ──
            proj["risk_score"] = min(1, max(0, (
                0.30 * proj["conflict_intensity"] +
                0.25 * proj["flood_risk"] +
                0.25 * proj["food_insecurity"] +
                0.20 * (proj["fragile_state_idx"] / 120)
            )))
            
            # ── Apply scenario-based growth rate to displacement ──
            proj["displaced_persons"] = int(base_displaced * ((1 + growth_rate) ** i))
            
            rows.append(proj)

    future_df = pd.DataFrame(rows)
    
    # Classify risk using the ML model
    X_f = _scaler.transform(future_df[features])
    future_df["risk_class"] = _clf.predict(X_f)

    return future_df


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
df = generate_dataset()
clf, reg, scaler, features, cv_scores, importances = train_models(df)

RISK_COLORS = {"Critical": "#ff4b4b", "High": "#ff8c00", "Medium": "#ffd700", "Low": "#00cc88"}
RISK_ORDER   = ["Critical", "High", "Medium", "Low"]


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Refugee Risk Predictor")
    st.markdown("---")

    selected_year = st.slider("📅 Year", 2010, 2030, 2025)
    selected_regions = st.multiselect(
        "🗺 Region Filter",
        options=sorted(df["region"].unique()),
        default=sorted(df["region"].unique())
    )
    selected_risk = st.multiselect(
        "⚠️ Risk Level Filter",
        options=RISK_ORDER,
        default=RISK_ORDER
    )
    
    st.markdown("---")
    st.markdown("### 📈 Forecast Scenario")
    selected_scenario = st.selectbox(
        "Select projection scenario",
        options=list(SCENARIOS.keys()),
        index=1  # Default to Moderate
    )
    
    show_forecast = st.checkbox("Show 2026–2030 Forecast", value=True)

    st.markdown("---")
    st.markdown("### 🧪 Model Accuracy")
    st.metric("CV Accuracy (5-fold)", f"{cv_scores.mean():.1%}", f"±{cv_scores.std():.1%}")

    st.markdown("---")
    st.markdown("### 📊 Methodology")
    st.caption("This dashboard uses **synthetic data** designed to mirror the structure and patterns of real-world humanitarian datasets.")
    st.caption("**Indicator categories modeled after:**")
    st.caption("• Climate: IPCC, EM-DAT")
    st.caption("• Conflict: ACLED, UCDP")
    st.caption("• Economic: World Bank WDI, FAO")
    st.caption("• Governance: Fragile States Index, Transparency International")
    st.markdown("---")
    st.warning("⚠️ For demonstration and research purposes. Not based on actual source data.")


# ─────────────────────────────────────────────
# GENERATE FORECAST WITH SELECTED SCENARIO
# ─────────────────────────────────────────────
growth_rate = SCENARIOS[selected_scenario]["rate"]
future_df = forecast_future(df, clf, reg, scaler, features, growth_rate)
full_df = pd.concat([df, future_df], ignore_index=True)


# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
display_df = full_df[
    (full_df["year"] == selected_year) &
    (full_df["region"].isin(selected_regions)) &
    (full_df["risk_class"].isin(selected_risk))
].copy()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 🌍 Refugee & Displacement Risk Predictor")
st.markdown(f"**Year: {selected_year}** {'🔮 Forecast' if selected_year > 2025 else '📊 Historical/Current'} | Showing {len(display_df)} countries")

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

total_displaced = display_df["displaced_persons"].sum()
critical_count  = (display_df["risk_class"] == "Critical").sum()
high_count      = (display_df["risk_class"] == "High").sum()
avg_risk        = display_df["risk_score"].mean()
climate_driven  = display_df[display_df["flood_risk"] > 0.65]["displaced_persons"].sum()

with k1:
    st.metric("🏃 Total Displaced", f"{total_displaced/1e6:.1f}M")
with k2:
    st.metric("🔴 Critical Risk Countries", str(critical_count))
with k3:
    st.metric("🟠 High Risk Countries", str(high_count))
with k4:
    st.metric("📈 Avg Risk Score", f"{avg_risk:.2f}")
with k5:
    st.metric("🌊 Climate-Driven Displaced", f"{climate_driven/1e6:.1f}M")

st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺 Risk Map", "📈 Forecasts", "🔍 Country Deep Dive",
    "⚙️ Model Insights", "📋 Full Data Table"
])

# ── TAB 1: WORLD RISK MAP ──
with tab1:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig_map = px.choropleth(
            display_df,
            locations="country",
            locationmode="country names",
            color="risk_score",
            hover_name="country",
            hover_data={
                "risk_class": True,
                "displaced_persons": ":,",
                "conflict_intensity": ":.2f",
                "flood_risk": ":.2f",
                "food_insecurity": ":.2f",
                "risk_score": ":.3f"
            },
            color_continuous_scale=[
                [0.0, "#00cc88"], [0.35, "#ffd700"],
                [0.55, "#ff8c00"], [0.75, "#ff4b4b"], [1.0, "#8b0000"]
            ],
            range_color=[0, 1],
            title=f"Global Displacement Risk Score — {selected_year}",
            template="plotly_dark"
        )
        fig_map.update_layout(
            height=440, margin=dict(l=0, r=0, t=40, b=0),
            coloraxis_colorbar=dict(title="Risk Score")
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_b:
        risk_counts = display_df.groupby("risk_class")["displaced_persons"].agg(
            Countries="count", Displaced="sum"
        ).reset_index()
        risk_counts["risk_class"] = pd.Categorical(risk_counts["risk_class"], categories=RISK_ORDER, ordered=True)
        risk_counts = risk_counts.sort_values("risk_class")

        fig_donut = px.pie(
            risk_counts, names="risk_class", values="Countries",
            color="risk_class",
            color_discrete_map=RISK_COLORS,
            hole=0.55,
            title="Countries by Risk Class"
        )
        fig_donut.update_layout(height=210, template="plotly_dark", margin=dict(t=40, b=0))
        st.plotly_chart(fig_donut, use_container_width=True)

        fig_bar = px.bar(
            risk_counts, x="risk_class", y="Displaced",
            color="risk_class", color_discrete_map=RISK_COLORS,
            title="Total Displaced by Risk Class",
            labels={"Displaced": "Displaced Persons", "risk_class": ""},
            template="plotly_dark"
        )
        fig_bar.update_layout(height=210, showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

    # Top 15 countries
    top15 = display_df.nlargest(15, "displaced_persons")
    fig_top = px.bar(
        top15, x="displaced_persons", y="country", orientation="h",
        color="risk_class", color_discrete_map=RISK_COLORS,
        title=f"Top 15 Countries by Displaced Persons ({selected_year})",
        labels={"displaced_persons": "Displaced Persons", "country": ""},
        template="plotly_dark"
    )
    fig_top.update_layout(height=380, showlegend=True, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_top, use_container_width=True)


# ── TAB 2: FORECASTS ──
with tab2:
    st.subheader("📈 Displacement Forecasts (2010–2030)")
    
    # Scenario explanation box
    scenario_info = SCENARIOS[selected_scenario]
    st.info(f"**{selected_scenario}**\n\n{scenario_info['description']}")

    top_countries_for_forecast = (
        df[df["year"] == 2025]
        .nlargest(10, "displaced_persons")["country"].tolist()
    )

    sel_countries = st.multiselect(
        "Select countries to compare",
        options=sorted(df["country"].unique()),
        default=top_countries_for_forecast[:6]
    )

    if sel_countries:
        trend_df = full_df[full_df["country"].isin(sel_countries)].copy()

        fig_trend = px.line(
            trend_df, x="year", y="displaced_persons",
            color="country", line_dash="country",
            title=f"Displacement Trend + Forecast ({selected_scenario})",
            labels={"displaced_persons": "Displaced Persons", "year": "Year"},
            template="plotly_dark"
        )
        fig_trend.add_vrect(x0=2025.5, x1=2030, fillcolor="#ffffff", opacity=0.04,
                            annotation_text="Forecast →", annotation_position="top left")
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)

        # Stacked area: risk class breakdown over time globally
        risk_time = full_df.groupby(["year", "risk_class"])["displaced_persons"].sum().reset_index()
        risk_time["risk_class"] = pd.Categorical(risk_time["risk_class"], categories=RISK_ORDER, ordered=True)
        risk_time = risk_time.sort_values(["year", "risk_class"])

        fig_area = px.area(
            risk_time, x="year", y="displaced_persons", color="risk_class",
            color_discrete_map=RISK_COLORS,
            title="Global Displaced Persons by Risk Class Over Time",
            labels={"displaced_persons": "Displaced Persons", "year": "Year"},
            template="plotly_dark"
        )
        fig_area.add_vrect(x0=2025.5, x1=2030, fillcolor="#ffffff", opacity=0.04,
                           annotation_text="Forecast", annotation_position="top left")
        fig_area.update_layout(height=350)
        st.plotly_chart(fig_area, use_container_width=True)

    # Methodology box
    st.markdown("---")
    st.subheader("📊 Forecast Methodology")
    
    st.markdown("""
    Projections are based on **UNHCR Global Trends data** showing displacement nearly doubled over the past decade.
    
    | Year | Growth | Key Driver |
    |------|--------|------------|
    | 2022 | **+21%** | Ukraine crisis (largest single-year increase on record) |
    | 2023 | **+8%** | Sudan civil war began |
    | 2024 | **+6%** | Sudan crisis deepened; 123.2M displaced globally |
    | Early 2025 | **−1%** | Syrian refugee returns (first decrease in over a decade) |
    
    **Scenario Definitions:**
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("**Conservative (3%)**\n\nPeace in Syria, diplomatic progress in Sudan, stable South Asia, no new conflicts.")
    
    with col2:
        st.warning("**Moderate (6%)**\n\nCurrent crises persist — Sudan, DRC, Myanmar, Afghanistan. No major escalation or resolution.")
    
    with col3:
        st.error("**Pessimistic (8%)**\n\nEscalation in existing hotspots or new conflicts — Afghanistan-Pakistan, India-Pakistan, Ethiopia, major climate disasters.")
    
    st.markdown("---")
    st.markdown("**Key Displacement Risk Factors:**")
    st.markdown("""
    - **Conflict:** Sudan civil war, Myanmar junta, DRC armed groups, Sahel insurgencies, Ukraine war
    - **Regional tensions:** Afghanistan-Pakistan border clashes, India-Pakistan (Kashmir), Ethiopia-Eritrea
    - **Climate:** Bangladesh flooding, Horn of Africa drought, Pacific island sea-level rise, Central American hurricanes
    - **Governance collapse:** Haiti gang violence, Venezuela economic migration, Lebanon financial crisis
    
    *Source: UNHCR Global Trends Report 2024*
    """)


# ── TAB 3: COUNTRY DEEP DIVE ──
with tab3:
    st.subheader("🔍 Country Deep Dive")
    country_sel = st.selectbox("Select Country", sorted(df["country"].unique()), index=0)
    c_data = full_df[full_df["country"] == country_sel].sort_values("year")

    latest = c_data[c_data["year"] == min(selected_year, 2025)].iloc[-1]

    rc = latest["risk_class"]
    color = RISK_COLORS.get(rc, "#aaa")
    st.markdown(
        f"<div class='metric-card'><span style='font-size:1.3rem; color:{color}'>⚠️ Risk Level: {rc}</span>"
        f" &nbsp;|&nbsp; Risk Score: <b>{latest['risk_score']:.3f}</b>"
        f" &nbsp;|&nbsp; Region: <b>{latest['region']}</b></div>",
        unsafe_allow_html=True
    )
    st.markdown("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Displaced Persons", f"{int(latest['displaced_persons']):,}")
    m2.metric("Conflict Intensity", f"{latest['conflict_intensity']:.2f}")
    m3.metric("Flood Risk", f"{latest['flood_risk']:.2f}")
    m4.metric("Food Insecurity", f"{latest['food_insecurity']:.2f}")

    # Multi-indicator radar
    categories = ["Conflict", "Climate", "Food Security", "Governance Fragility", "Economic Stress"]
    vals = [
        latest["conflict_intensity"],
        latest["flood_risk"],
        latest["food_insecurity"],
        latest["fragile_state_idx"] / 120,
        latest["gini_coeff"] / 70
    ]

    fig_radar = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255, 75, 75, 0.3)",
        line=dict(color="#ff4b4b", width=2),
        name=country_sel
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True, color="#888")),
        title=f"{country_sel} — Risk Radar", template="plotly_dark", height=380
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Time series for this country
    fig_c1, fig_c2 = st.columns(2)
    with fig_c1:
        fig_disp = px.area(c_data, x="year", y="displaced_persons",
            title="Displaced Persons Over Time",
            template="plotly_dark", color_discrete_sequence=["#ff6b6b"])
        fig_disp.add_vrect(x0=2025.5, x1=2030, fillcolor="#ffffff", opacity=0.05,
                           annotation_text="Forecast")
        fig_disp.update_layout(height=260)
        st.plotly_chart(fig_disp, use_container_width=True)

    with fig_c2:
        fig_risk_ts = px.line(c_data, x="year", y="risk_score",
            title="Risk Score Over Time",
            template="plotly_dark", color_discrete_sequence=["#ffd700"])
        fig_risk_ts.add_vrect(x0=2025.5, x1=2030, fillcolor="#ffffff", opacity=0.05,
                              annotation_text="Forecast")
        fig_risk_ts.update_layout(height=260)
        st.plotly_chart(fig_risk_ts, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        fig_conf = px.line(c_data, x="year", y=["conflict_intensity", "gov_territory_loss", "battle_deaths"],
            title="Conflict Indicators", template="plotly_dark")
        fig_conf.update_layout(height=240)
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_right:
        fig_clim = px.line(c_data, x="year", y=["temp_anomaly", "flood_risk", "drought_index", "crop_yield_loss"],
            title="Climate Indicators", template="plotly_dark")
        fig_clim.update_layout(height=240)
        st.plotly_chart(fig_clim, use_container_width=True)


# ── TAB 4: MODEL INSIGHTS ──
with tab4:
    st.subheader("⚙️ Model Insights & Feature Importance")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Classifier", "Random Forest (n=200)")
        st.metric("Regressor", "Gradient Boosting (n=200)")
        st.metric("Training Period", "2010–2023")
        st.metric("Countries", "50")
        st.metric("Features", str(len(features)))
        st.metric("5-Fold CV Accuracy", f"{cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    with col2:
        fig_imp = px.bar(
            importances.reset_index().rename(columns={"index": "Feature", 0: "Importance"}),
            x="Importance", y="Feature", orientation="h",
            title="Feature Importance (Random Forest Classifier)",
            color="Importance", color_continuous_scale="Reds",
            template="plotly_dark"
        )
        fig_imp.update_layout(height=520, yaxis={"categoryorder": "total ascending"}, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("📐 Feature Correlation with Risk Score")
    corr_data = full_df[features + ["risk_score"]].corr()["risk_score"].drop("risk_score").sort_values()
    fig_corr = px.bar(
        corr_data.reset_index().rename(columns={"index": "Feature", "risk_score": "Correlation"}),
        x="Correlation", y="Feature", orientation="h",
        color="Correlation",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation with Risk Score",
        template="plotly_dark"
    )
    fig_corr.update_layout(height=460, yaxis={"categoryorder": "total ascending"}, showlegend=False)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("🧩 Methodology Notes")
    st.markdown("""
    | Component | Method | Notes |
    |---|---|---|
    | Risk Classification | Random Forest (200 trees) | 4-class: Critical / High / Medium / Low |
    | Displacement Volume | Scenario-based projection | 3% / 6% / 8% annual growth rates |
    | Forecast Horizon | 2026–2030 | Based on UNHCR historical growth rates |
    | Class Imbalance | Stratified CV | No SMOTE required at 50-country scale |
    | Data normalization | StandardScaler | Applied before both models |
    
    **Driver categories and weights:**
    - 🔴 Conflict indicators (ACLED events, battle deaths, territory loss): **30%**
    - 🌡 Climate indicators (temp anomaly, flood risk, drought, sea level): **25%**
    - 💰 Economic indicators (GDP, food insecurity, Gini): **25%**
    - 🏛 Governance indicators (FSI, press freedom, rule of law): **20%**
    
    ---
    
    **⚠️ Data Disclaimer:** This dashboard uses synthetic data designed to mirror the structure and patterns 
    of real-world humanitarian datasets. It is intended for demonstration and research purposes only 
    and is not based on actual source data.
    """)


# ── TAB 5: DATA TABLE ──
with tab5:
    st.subheader(f"📋 Full Data — {selected_year}")

    view_cols = ["country", "region", "risk_class", "risk_score", "displaced_persons",
                 "conflict_intensity", "flood_risk", "food_insecurity",
                 "gdp_per_capita", "fragile_state_idx"]

    styled = display_df[view_cols].sort_values("risk_score", ascending=False).reset_index(drop=True)
    styled["displaced_persons"] = styled["displaced_persons"].apply(lambda x: f"{x:,}")
    styled["risk_score"]        = styled["risk_score"].apply(lambda x: f"{x:.3f}")
    styled["gdp_per_capita"]    = styled["gdp_per_capita"].apply(lambda x: f"${float(x):,.0f}")

    st.dataframe(styled, use_container_width=True, height=600)

    csv_out = display_df.to_csv(index=False)
    st.download_button("⬇️ Download Full Dataset (CSV)", csv_out,
                       file_name=f"refugee_risk_{selected_year}.csv", mime="text/csv")
