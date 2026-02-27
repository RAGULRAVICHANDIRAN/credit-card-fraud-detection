"""
🛡️ Fraud Detection Dashboard — Stunning Animated Version
Launch:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import time
import datetime

from dashboard.report_generator import generate_risk_report, _find_similar_transactions

from src.utils.config import (
    MODELS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR,
    DATASET_NAME, RANDOM_SEED, ALL_STRATEGIES,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🛡️ FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# PREMIUM CSS — Dark theme, glassmorphism, animations
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global dark theme ── */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a3e 40%, #0d0d2b 100%);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f2e 0%, #1a1040 100%) !important;
        border-right: 1px solid rgba(100, 100, 255, 0.15);
    }

    [data-testid="stSidebar"] * {
        color: #e0e0f0 !important;
    }

    /* ── Glass cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px;
        margin: 10px 0;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: slideUp 0.6s ease-out;
    }
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(100, 100, 255, 0.3);
        box-shadow: 0 20px 60px rgba(80, 60, 255, 0.15);
    }

    /* ── Metric cards with glow ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.2));
        backdrop-filter: blur(20px);
        border: 1px solid rgba(139, 92, 246, 0.25);
        border-radius: 20px;
        padding: 24px 20px;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        animation: fadeInScale 0.5s ease-out backwards;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 60%);
        animation: shimmer 3s ease-in-out infinite;
    }
    .metric-card:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 0 25px 50px rgba(139, 92, 246, 0.25);
        border-color: rgba(139, 92, 246, 0.5);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }
    .metric-label {
        font-size: 0.85rem;
        color: rgba(200, 200, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-top: 6px;
        position: relative;
        z-index: 1;
    }
    .metric-icon {
        font-size: 1.6rem;
        margin-bottom: 6px;
        position: relative;
        z-index: 1;
    }

    /* ── Fraud / Legit indicator cards ── */
    .fraud-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.15));
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 20px;
        padding: 24px;
        animation: pulseGlow 2s ease-in-out infinite;
    }
    .legit-card {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15));
        border: 1px solid rgba(34, 197, 94, 0.3);
        border-radius: 20px;
        padding: 24px;
        animation: fadeInScale 0.5s ease-out;
    }

    /* ── Title styles ── */
    .page-title {
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
        animation: fadeInScale 0.6s ease-out;
    }
    .page-subtitle {
        font-size: 1.1rem;
        color: rgba(200, 200, 255, 0.6);
        font-weight: 400;
        margin-bottom: 30px;
        animation: slideUp 0.8s ease-out;
    }

    /* ── Sidebar styling ── */
    .sidebar-title {
        font-size: 1.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 20px;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        animation: fadeInScale 0.4s ease-out;
    }
    .badge-fraud {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-safe {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    /* ── Animations ── */
    @keyframes fadeInScale {
        from { opacity: 0; transform: scale(0.9); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes shimmer {
        0%, 100% { transform: rotate(0deg); }
        50%      { transform: rotate(180deg); }
    }
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.15); }
        50%      { box-shadow: 0 0 40px rgba(239, 68, 68, 0.3); }
    }
    @keyframes countUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ── Delay helpers ── */
    .delay-1 { animation-delay: 0.1s; }
    .delay-2 { animation-delay: 0.2s; }
    .delay-3 { animation-delay: 0.3s; }
    .delay-4 { animation-delay: 0.4s; }

    /* ── Plotly chart backgrounds ── */
    .stPlotlyChart {
        border-radius: 16px;
        overflow: hidden;
    }

    /* ── Divider ── */
    .glow-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.5), transparent);
        border: none;
        margin: 30px 0;
        animation: slideUp 0.8s ease-out;
    }

    /* headings & text */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #e0e0f0 !important;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: rgba(200, 200, 255, 0.8) !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Plotly dark template
# ──────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
CHART_COLORS = ["#818cf8", "#f472b6", "#34d399", "#fbbf24", "#f87171",
                "#a78bfa", "#38bdf8", "#fb923c", "#4ade80", "#e879f9"]
CHART_BG = "rgba(0,0,0,0)"
PAPER_BG = "rgba(0,0,0,0)"


def styled_plotly(fig, height=450):
    """Apply consistent dark styling to plotly figures."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=CHART_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(family="Inter", color="#c0c0e0"),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    )
    return fig


# ──────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────

@st.cache_data
def load_results():
    path = MODELS_DIR / "existing_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dataset_sample(n=10000):
    try:
        d = PROCESSED_DATA_DIR / DATASET_NAME
        X = pd.read_csv(d / "X_test.csv")
        y = pd.read_csv(d / "y_test.csv").squeeze()
        df = X.copy()
        df["target"] = y.values
        return df.head(n)
    except FileNotFoundError:
        return pd.DataFrame()


# Feature descriptions for explanations
_FEATURE_LABELS = {
    "Amount": "Transaction amount",
    "Time": "Transaction time",
}

def _describe_feature(feat_name, value, impact):
    label = _FEATURE_LABELS.get(feat_name, feat_name)
    if impact > 0:
        risk = "⬆️ higher"
    else:
        risk = "⬇️ lower"

    if feat_name == "Amount":
        if impact > 0:
            return f"💰 **{label}** (${value:.2f}) is unusually high — large transactions are a common fraud signal."
        return f"💰 **{label}** (${value:.2f}) is within normal range — typical of legitimate transactions."
    if feat_name == "Time":
        if impact > 0:
            return f"🕐 **{label}** ({value:.0f}s) falls in an unusual time window — fraud often occurs at odd hours."
        return f"🕐 **{label}** ({value:.0f}s) is during normal hours — typical of legitimate transactions."
    if feat_name.startswith("V") and feat_name[1:].isdigit():
        n = feat_name[1:]
        return f"🔬 **Hidden pattern {n}** (value: {value:.3f}) — {risk} fraud risk based on spending behavior analysis."
    strength = "strongly" if abs(impact) > 1.0 else "slightly"
    return f"📊 **{label}** (value: {value:.3f}) {strength} {risk} fraud risk."


# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════

st.sidebar.markdown('<div class="sidebar-title">🛡️ FraudShield AI</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📊 Data Explorer", "📈 Model Arena",
     "🔮 Predict & Explain", "🔐 2FA Simulator", "⚡ Live Monitor"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="text-align:center; color:rgba(200,200,255,0.4); font-size:0.75rem;">'
    'Built with ❤️ by RAGUL AND ITS TEAM<br>'
    'A PROJECT BY SGI BOYS<br><br>'
    'RAGUL. R<br>ANISHKUMAR. P<br>ROBERTCHRISTOPHER. A</div>',
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════
# PAGE 1 — Dashboard (Overview)
# ══════════════════════════════════════════════

if page == "🏠 Dashboard":
    st.markdown('<div class="page-title">🏠 Command Center</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Real-time overview of fraud detection performance</div>', unsafe_allow_html=True)

    df_sample = load_dataset_sample()
    df_results = load_results()

    if df_sample.empty:
        st.warning("⚠️ No data found. Run `python main.py preprocess` first.")
    else:
        n_total = len(df_sample)
        n_fraud = int(df_sample["target"].sum())
        n_legit = n_total - n_fraud
        fraud_pct = n_fraud / n_total * 100

        # Animated metric cards
        cols = st.columns(4)
        metrics_data = [
            ("📦", f"{n_total:,}", "Total Transactions"),
            ("✅", f"{n_legit:,}", "Legitimate"),
            ("🚨", f"{n_fraud:,}", "Fraudulent"),
            ("📊", f"{fraud_pct:.3f}%", "Fraud Rate"),
        ]
        for i, (icon, val, label) in enumerate(metrics_data):
            cols[i].markdown(f"""
                <div class="metric-card delay-{i+1}">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # Two charts side by side
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            # Animated donut chart
            fig = go.Figure(go.Pie(
                labels=["Legitimate", "Fraudulent"],
                values=[n_legit, n_fraud],
                hole=0.65,
                marker=dict(colors=["#34d399", "#f87171"],
                            line=dict(color="#0a0a1a", width=3)),
                textinfo="percent+label",
                textfont=dict(size=14, family="Inter"),
                pull=[0, 0.08],
            ))
            fig = styled_plotly(fig, height=380)
            fig.update_layout(
                title=dict(text="Class Distribution", font=dict(size=18)),
                showlegend=False,
                annotations=[dict(text=f"<b>{fraud_pct:.2f}%</b><br>Fraud",
                                  x=0.5, y=0.5, font_size=20, showarrow=False,
                                  font=dict(color="#f472b6"))],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            # Transaction amount distribution
            fig = go.Figure()
            for cls, name, color in [(0, "Legitimate", "#34d399"), (1, "Fraudulent", "#f87171")]:
                subset = df_sample[df_sample["target"] == cls]["Amount"]
                fig.add_trace(go.Histogram(
                    x=subset, name=name, marker_color=color,
                    opacity=0.7, nbinsx=50,
                ))
            fig = styled_plotly(fig, height=380)
            fig.update_layout(
                title=dict(text="Amount Distribution", font=dict(size=18)),
                barmode="overlay",
                xaxis_title="Amount (scaled)",
                yaxis_title="Count",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Best model performance
        if not df_results.empty:
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🏆 Top Model Performance")

            best = df_results.sort_values("f1", ascending=False).head(5)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=best["model"] + " / " + best["strategy"],
                y=best["f1"],
                marker=dict(
                    color=best["f1"],
                    colorscale=[[0, "#818cf8"], [0.5, "#c084fc"], [1, "#f472b6"]],
                    line=dict(width=0),
                    cornerradius=8,
                ),
                text=[f"{v:.4f}" for v in best["f1"]],
                textposition="outside",
                textfont=dict(color="#c0c0e0", size=13),
            ))
            fig = styled_plotly(fig, height=350)
            fig.update_layout(
                title=dict(text="Top 5 F1 Scores", font=dict(size=18)),
                yaxis=dict(range=[0, 1.05]),
                xaxis_title="", yaxis_title="F1 Score",
            )
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 — Data Explorer
# ══════════════════════════════════════════════

elif page == "📊 Data Explorer":
    st.markdown('<div class="page-title">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Deep dive into the European Cardholders dataset</div>', unsafe_allow_html=True)

    df = load_dataset_sample(n=20000)

    if df.empty:
        st.warning("No data found.")
    else:
        # Feature correlation heatmap
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 Feature Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(
            corr, color_continuous_scale="Purples",
            aspect="auto",
        )
        fig = styled_plotly(fig, height=500)
        fig.update_layout(title="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # PCA feature distributions
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📐 Feature Distributions (Fraud vs Legitimate)")
        feat_choice = st.selectbox("Select feature", [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"])

        fig = go.Figure()
        for cls, name, color in [(0, "Legitimate", "#34d399"), (1, "Fraudulent", "#f87171")]:
            subset = df[df["target"] == cls][feat_choice]
            fig.add_trace(go.Histogram(
                x=subset, name=name, marker_color=color,
                opacity=0.6, nbinsx=80,
            ))
        fig = styled_plotly(fig, height=380)
        fig.update_layout(barmode="overlay", xaxis_title=feat_choice, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Stats
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Descriptive Statistics")
        st.dataframe(
            df.describe().T.style.format("{:.3f}").set_properties(**{
                'background-color': 'rgba(0,0,0,0.3)',
                'color': '#c0c0e0',
            }),
            use_container_width=True, height=400,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 3 — Model Arena
# ══════════════════════════════════════════════

elif page == "📈 Model Arena":
    st.markdown('<div class="page-title">📈 Model Arena</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Compare all models head-to-head across balancing strategies</div>', unsafe_allow_html=True)

    df = load_results()
    if df.empty:
        st.warning("No results found. Run training first.")
    else:
        # Results table
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📋 Full Results Table")
        st.dataframe(
            df.style.format({
                "accuracy": "{:.4f}", "precision": "{:.4f}",
                "recall": "{:.4f}", "f1": "{:.4f}", "roc_auc": "{:.4f}",
            }).background_gradient(subset=["f1", "roc_auc"], cmap="RdYlGn"),
            use_container_width=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        # Animated grouped bar chart
        metric = st.selectbox("📊 Metric to visualize", ["f1", "roc_auc", "precision", "recall", "accuracy"])

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = go.Figure()
        for i, strat in enumerate(df["strategy"].unique()):
            sub = df[df["strategy"] == strat]
            fig.add_trace(go.Bar(
                name=strat,
                x=sub["model"],
                y=sub[metric],
                marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                text=[f"{v:.3f}" for v in sub[metric]],
                textposition="outside",
                textfont=dict(size=11),
            ))
        fig = styled_plotly(fig, height=420)
        fig.update_layout(
            barmode="group",
            title=dict(text=f"{metric.upper()} by Model & Strategy", font=dict(size=18)),
            yaxis=dict(range=[0, 1.1]),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Radar chart
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Model Radar Comparison")
        radar_metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        best_per_model = df.groupby("model")[radar_metrics].max().reset_index()

        fig = go.Figure()
        for i, row in best_per_model.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics] + [row[radar_metrics[0]]],
                theta=[m.upper() for m in radar_metrics] + [radar_metrics[0].upper()],
                fill="toself",
                name=row["model"],
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
                opacity=0.7,
            ))
        fig = styled_plotly(fig, height=450)
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(200,200,255,0.1)"),
                angularaxis=dict(gridcolor="rgba(200,200,255,0.1)"),
            ),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Heatmap
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🗺️ Performance Heatmap")
        pivot = df.pivot_table(values=metric, index="strategy", columns="model")
        fig = px.imshow(
            pivot, text_auto=".3f",
            color_continuous_scale=[[0, "#1e1b4b"], [0.5, "#7c3aed"], [1, "#f472b6"]],
            aspect="auto",
        )
        fig = styled_plotly(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Explainability image gallery
        fig_dir = FIGURES_DIR / DATASET_NAME
        if fig_dir.exists():
            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🔍 Explainability Plots (SHAP & LIME)")
            images = sorted(fig_dir.glob("*.png"))
            if images:
                img_cols = st.columns(min(len(images), 3))
                for i, img in enumerate(images):
                    img_cols[i % 3].image(str(img), caption=img.stem.replace("_", " ").title(),
                                          use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 4 — Predict & Explain
# ══════════════════════════════════════════════

elif page == "🔮 Predict & Explain":
    st.markdown('<div class="page-title">🔮 Predict & Explain</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Enter transaction features to get a prediction with plain-English explanation</div>', unsafe_allow_html=True)

    # Only show the 4 core ML models (one per type)
    _MODEL_MAP = {
        "Naive Bayes": "naive_bayes",
        "Random Forest": "random_forest",
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
    }
    # Find one saved file per model type (prefer SMOTE strategy)
    _available = {}
    for display, key in _MODEL_MAP.items():
        # Try SMOTE version first, then any match
        smote = MODELS_DIR / f"{DATASET_NAME}_smote_{key}.joblib"
        if smote.exists():
            _available[display] = smote
        else:
            matches = list(MODELS_DIR.glob(f"*_{key}.joblib"))
            if matches:
                _available[display] = matches[0]

    # Add ensemble option if at least 2 models are available
    if len(_available) >= 2:
        _available["🏆 Ensemble (All Models)"] = "ensemble"

    if not _available:
        st.warning("No saved models found.")
    else:
        model_choice = st.selectbox("🤖 Select Model", list(_available.keys()))

        sample_df = load_dataset_sample(n=1)
        if not sample_df.empty:
            feature_cols = [c for c in sample_df.columns if c != "target"]

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(f"### ⚙️ Input Features ({len(feature_cols)} features)")
            cols = st.columns(4)
            values = {}
            for i, feat in enumerate(feature_cols):
                with cols[i % 4]:
                    values[feat] = st.number_input(feat, value=0.0, format="%.4f", key=f"f_{feat}")
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
                # Animated spinner
                with st.spinner("🔄 Analyzing transaction patterns..."):
                    time.sleep(0.5)

                input_df = pd.DataFrame([values])

                if model_choice == "🏆 Ensemble (All Models)":
                    # ── Ensemble: majority vote + averaged probabilities ──
                    individual = {k: v for k, v in _available.items() if v != "ensemble"}
                    all_preds = {}
                    all_probs = {}
                    for name, path in individual.items():
                        m = joblib.load(path)
                        all_preds[name] = int(m.predict(input_df)[0])
                        if hasattr(m, "predict_proba"):
                            all_probs[name] = m.predict_proba(input_df)[0]

                    # Majority vote
                    votes_fraud = sum(v for v in all_preds.values())
                    pred = 1 if votes_fraud > len(all_preds) / 2 else 0

                    # Average probabilities
                    if all_probs:
                        avg_prob = np.mean([p for p in all_probs.values()], axis=0)
                        prob = avg_prob
                    else:
                        prob = None

                    # Show per-model votes
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown("### 🗳️ Per-Model Votes")
                    vote_cols = st.columns(len(all_preds))
                    for i, (name, v) in enumerate(all_preds.items()):
                        emoji = "🚨" if v == 1 else "✅"
                        label = "FRAUD" if v == 1 else "SAFE"
                        color = "#fca5a5" if v == 1 else "#86efac"
                        conf_str = ""
                        if name in all_probs:
                            c = all_probs[name][1] * 100 if v == 1 else all_probs[name][0] * 100
                            conf_str = f"<br><span style='font-size:0.8rem;'>{c:.1f}%</span>"
                        vote_cols[i].markdown(f"""
                            <div class="metric-card delay-{i+1}">
                                <div class="metric-icon">{emoji}</div>
                                <div style="font-size:1rem; font-weight:700; color:{color};">{label}</div>
                                <div class="metric-label">{name}{conf_str}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                    # Use first tree-based model for SHAP explanation
                    model = None
                    for name in ["XGBoost", "Random Forest", "LightGBM"]:
                        if name in individual:
                            model = joblib.load(individual[name])
                            break
                    if model is None:
                        model = joblib.load(list(individual.values())[0])
                else:
                    # ── Single model ──
                    model_path = _available[model_choice]
                    model = joblib.load(model_path)
                    pred = model.predict(input_df)[0]
                    prob = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else None

                # Result card
                if pred == 1:
                    conf = f"{prob[1]*100:.1f}%" if prob is not None else "N/A"
                    st.markdown(f"""
                        <div class="fraud-card">
                            <div style="text-align:center;">
                                <div style="font-size:3rem; margin-bottom:10px;">🚨</div>
                                <div style="font-size:1.8rem; font-weight:800; color:#fca5a5;">
                                    FRAUDULENT TRANSACTION
                                </div>
                                <div style="font-size:1.1rem; color:rgba(252,165,165,0.7); margin-top:8px;">
                                    Confidence: {conf}
                                </div>
                                <div class="status-badge badge-fraud" style="margin-top:12px;">
                                    ⚠️ HIGH RISK — RECOMMEND BLOCKING
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    conf = f"{prob[0]*100:.1f}%" if prob is not None else "N/A"
                    st.markdown(f"""
                        <div class="legit-card">
                            <div style="text-align:center;">
                                <div style="font-size:3rem; margin-bottom:10px;">✅</div>
                                <div style="font-size:1.8rem; font-weight:800; color:#86efac;">
                                    LEGITIMATE TRANSACTION
                                </div>
                                <div style="font-size:1.1rem; color:rgba(134,239,172,0.7); margin-top:8px;">
                                    Confidence: {conf}
                                </div>
                                <div class="status-badge badge-safe" style="margin-top:12px;">
                                    ✅ LOW RISK — APPROVE
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                # Explanation
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📝 Why this decision?")

                explanation_done = False
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(input_df)
                    if isinstance(sv, list):
                        sv = sv[1][0]
                    else:
                        sv = sv[0]
                    contrib = pd.Series(sv, index=feature_cols)
                    top = contrib.abs().sort_values(ascending=False).head(5)

                    reasons = [_describe_feature(f, values[f], contrib[f]) for f in top.index]
                    for i, r in enumerate(reasons, 1):
                        st.markdown(f"{i}. {r}")

                    # SHAP waterfall-style bar chart
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    top_contrib = contrib[top.index].sort_values()
                    colors = ["#f87171" if v > 0 else "#34d399" for v in top_contrib.values]
                    fig = go.Figure(go.Bar(
                        x=top_contrib.values, y=top_contrib.index,
                        orientation="h", marker_color=colors,
                        text=[f"{v:+.4f}" for v in top_contrib.values],
                        textposition="outside",
                    ))
                    fig = styled_plotly(fig, height=300)
                    fig.update_layout(
                        title="Feature Impact (SHAP values)",
                        xaxis_title="Impact on prediction",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    explanation_done = True
                except Exception:
                    pass

                if not explanation_done:
                    full_df = load_dataset_sample(n=5000)
                    if not full_df.empty:
                        reasons = []
                        for feat in feature_cols[:30]:
                            col = full_df[feat]
                            mean, std = col.mean(), col.std()
                            if std > 0:
                                z = (values[feat] - mean) / std
                                if abs(z) > 2.0:
                                    reasons.append((abs(z), _describe_feature(feat, values[feat], z)))
                        reasons.sort(reverse=True)
                        for i, (_, r) in enumerate(reasons[:5], 1):
                            st.markdown(f"{i}. {r}")
                        if not reasons:
                            st.info("All values within normal ranges. Decision based on subtle feature interactions.")

                # ── PDF Risk Report Download ──
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                # Collect report data
                _report_confidence = None
                if prob is not None:
                    _report_confidence = prob[1] if pred == 1 else prob[0]

                # Get SHAP values for report (reuse if available)
                _report_shap = None
                try:
                    if 'contrib' in dir() or 'contrib' in locals():
                        _report_shap = contrib.values if hasattr(contrib, 'values') else None
                except Exception:
                    pass
                # Try computing if not yet available
                if _report_shap is None:
                    try:
                        import shap as _shap_mod
                        _exp = _shap_mod.TreeExplainer(model)
                        _sv = _exp.shap_values(input_df)
                        if isinstance(_sv, list):
                            _report_shap = _sv[1][0]
                        else:
                            _report_shap = _sv[0]
                    except Exception:
                        pass

                # Collect reasons
                _report_reasons = []
                if _report_shap is not None:
                    _r_contrib = pd.Series(_report_shap, index=feature_cols)
                    _r_top = _r_contrib.abs().sort_values(ascending=False).head(5)
                    _report_reasons = [_describe_feature(f, values[f], _r_contrib[f]) for f in _r_top.index]

                # Find similar transactions
                _sim_df = None
                try:
                    _full_data = load_dataset_sample(n=5000)
                    if not _full_data.empty:
                        _sim_df = _find_similar_transactions(values, _full_data, feature_cols, n=5)
                except Exception:
                    pass

                # Get per-model votes if ensemble
                _votes = None
                _probs = None
                if model_choice == "\U0001f3c6 Ensemble (All Models)":
                    try:
                        _votes = all_preds
                        _probs = all_probs
                    except Exception:
                        pass

                try:
                    pdf_bytes = generate_risk_report(
                        prediction=pred,
                        confidence=_report_confidence,
                        model_name=model_choice,
                        feature_values=values,
                        feature_cols=feature_cols,
                        reasons=_report_reasons,
                        shap_values=_report_shap,
                        per_model_votes=_votes,
                        per_model_probs=_probs,
                        similar_df=_sim_df,
                    )
                    st.download_button(
                        label="\U0001f4cb Download Risk Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"fraud_risk_report_{int(time.time())}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )
                except Exception as e:
                    st.error(f"Could not generate PDF report: {e}")


# ══════════════════════════════════════════════
# PAGE 5 — 2FA / MFA Simulator
# ══════════════════════════════════════════════

elif page == "🔐 2FA Simulator":
    st.markdown('<div class="page-title">🔐 2FA / MFA Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Simulate risk-based multi-factor authentication on transactions</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    risk_2fa = col1.slider("🔓 2FA threshold", 0.0, 1.0, 0.5, 0.05)
    risk_mfa = col2.slider("🔒 MFA threshold", 0.0, 1.0, 0.8, 0.05)
    n_txn = col3.number_input("📊 Transactions", 100, 10000, 1000, 100)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        with st.spinner("⚡ Running simulation..."):
            time.sleep(0.3)

        np.random.seed(RANDOM_SEED)
        risk_scores = np.random.beta(2, 5, n_txn)
        n_fraud = int(n_txn * 0.05)
        risk_scores[:n_fraud] = np.random.beta(5, 2, n_fraud)
        np.random.shuffle(risk_scores)

        layers = np.where(
            risk_scores < risk_2fa, "Standard",
            np.where(risk_scores < risk_mfa, "2FA (SMS/Email)", "MFA (Biometric)"),
        )

        l1 = (layers == "Standard").sum()
        l2 = (layers == "2FA (SMS/Email)").sum()
        l3 = (layers == "MFA (Biometric)").sum()

        # Animated metric cards
        cols = st.columns(3)
        for i, (icon, val, pct, label, color) in enumerate([
            ("🟢", l1, l1/n_txn*100, "Standard Auth", "#34d399"),
            ("🟡", l2, l2/n_txn*100, "2FA Required", "#fbbf24"),
            ("🔴", l3, l3/n_txn*100, "MFA Required", "#f87171"),
        ]):
            cols[i].markdown(f"""
                <div class="metric-card delay-{i+1}">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{val:,}</div>
                    <div class="metric-label">{label} ({pct:.1f}%)</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            sim_df = pd.DataFrame({"risk_score": risk_scores, "auth_layer": layers})
            fig = go.Figure()
            for layer, color in [("Standard", "#34d399"), ("2FA (SMS/Email)", "#fbbf24"), ("MFA (Biometric)", "#f87171")]:
                subset = sim_df[sim_df["auth_layer"] == layer]["risk_score"]
                fig.add_trace(go.Histogram(x=subset, name=layer, marker_color=color, opacity=0.7, nbinsx=40))
            fig = styled_plotly(fig, height=380)
            fig.update_layout(barmode="overlay", title="Risk Score Distribution", xaxis_title="Risk Score")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            fig = go.Figure(go.Pie(
                labels=["Standard", "2FA", "MFA"],
                values=[l1, l2, l3],
                hole=0.6,
                marker=dict(colors=["#34d399", "#fbbf24", "#f87171"],
                            line=dict(color="#0a0a1a", width=3)),
                textinfo="percent+label",
                pull=[0, 0.03, 0.06],
            ))
            fig = styled_plotly(fig, height=380)
            fig.update_layout(title="Authentication Layer Split", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 6 — Live Monitor (ATM Machine App)
# ══════════════════════════════════════════════

elif page == "⚡ Live Monitor":
    st.markdown('<div class="page-title">⚡ Live Fraud Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Interactive ATM simulation with real-time fraud detection</div>', unsafe_allow_html=True)

    # ── ATM-specific CSS ──
    st.markdown("""
    <style>
    /* ── ATM Machine Container ── */
    .atm-machine {
        background: linear-gradient(145deg, #1a1a3e 0%, #0d0d2b 50%, #1a1040 100%);
        border: 2px solid rgba(139, 92, 246, 0.3);
        border-radius: 30px;
        padding: 40px 30px;
        max-width: 520px;
        margin: 20px auto;
        box-shadow:
            0 0 60px rgba(139, 92, 246, 0.15),
            0 30px 80px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    .atm-machine::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #7c3aed, #f472b6, #7c3aed, #818cf8);
        border-radius: 31px;
        z-index: -1;
        opacity: 0.4;
        animation: borderGlow 4s ease-in-out infinite;
    }
    @keyframes borderGlow {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 0.6; }
    }

    /* ── ATM Screen ── */
    .atm-screen {
        background: linear-gradient(135deg, #0a1628 0%, #0f1d33 100%);
        border: 2px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 30px 24px;
        margin-bottom: 24px;
        min-height: 180px;
        box-shadow:
            inset 0 0 30px rgba(0, 0, 0, 0.5),
            0 0 20px rgba(56, 189, 248, 0.1);
        position: relative;
        overflow: hidden;
    }
    .atm-screen::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.5), transparent);
    }
    .atm-screen-title {
        font-size: 0.75rem;
        color: rgba(56, 189, 248, 0.7);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        font-weight: 600;
    }
    .atm-screen-main {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e0f0ff;
        text-align: center;
        margin: 12px 0;
    }
    .atm-screen-sub {
        font-size: 0.9rem;
        color: rgba(200, 220, 255, 0.6);
        text-align: center;
    }

    /* ── Card Slot ── */
    .atm-card-slot {
        background: linear-gradient(135deg, #0a0a1a, #151530);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 8px;
        height: 12px;
        width: 65%;
        margin: 0 auto 20px;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.6);
        position: relative;
        overflow: hidden;
    }
    .atm-card-slot::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 0;
        right: 0;
        height: 1px;
        background: rgba(139, 92, 246, 0.3);
    }

    /* ── Keypad ── */
    .atm-keypad {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        max-width: 280px;
        margin: 0 auto;
        padding: 16px;
    }
    .atm-key {
        background: linear-gradient(135deg, rgba(30, 30, 60, 0.9), rgba(20, 20, 50, 0.9));
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 12px;
        padding: 14px 8px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        color: #c0c0e0;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    .atm-key:hover {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.3), rgba(99, 102, 241, 0.3));
        border-color: rgba(139, 92, 246, 0.5);
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.2);
    }
    .atm-key-action {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.4), rgba(99, 102, 241, 0.3));
        color: #c084fc;
        font-size: 0.85rem;
    }
    .atm-key-cancel {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(220, 38, 38, 0.2));
        color: #fca5a5;
        font-size: 0.85rem;
    }
    .atm-key-enter {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(22, 163, 74, 0.2));
        color: #86efac;
        font-size: 0.85rem;
    }

    /* ── ATM Logo / Brand ── */
    .atm-brand {
        text-align: center;
        margin-bottom: 20px;
    }
    .atm-brand-name {
        font-size: 1.3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 4px;
    }
    .atm-brand-sub {
        font-size: 0.65rem;
        color: rgba(200, 200, 255, 0.4);
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    /* ── Receipt ── */
    .atm-receipt {
        background: linear-gradient(180deg, #fefce8 0%, #fef9c3 100%);
        border-radius: 4px;
        padding: 24px 20px;
        margin: 16px auto;
        max-width: 360px;
        font-family: 'Courier New', monospace;
        color: #1a1a2e;
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
        position: relative;
    }
    .atm-receipt::before {
        content: '';
        position: absolute;
        bottom: -8px;
        left: 0;
        right: 0;
        height: 8px;
        background: repeating-linear-gradient(
            135deg,
            transparent,
            transparent 4px,
            #fef9c3 4px,
            #fef9c3 8px
        );
    }
    .receipt-header {
        text-align: center;
        font-weight: bold;
        font-size: 1rem;
        border-bottom: 1px dashed #94a3b8;
        padding-bottom: 10px;
        margin-bottom: 10px;
    }
    .receipt-line {
        display: flex;
        justify-content: space-between;
        padding: 3px 0;
        font-size: 0.8rem;
    }
    .receipt-footer {
        text-align: center;
        font-size: 0.7rem;
        color: #64748b;
        margin-top: 12px;
        border-top: 1px dashed #94a3b8;
        padding-top: 10px;
    }

    /* ── Approved / Declined overlays ── */
    .atm-approved {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
        border: 2px solid rgba(34, 197, 94, 0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        animation: fadeInScale 0.5s ease-out;
    }
    .atm-declined {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1));
        border: 2px solid rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        animation: fadeInScale 0.5s ease-out, pulseGlow 2s ease-in-out infinite;
    }

    /* ── Transaction log ── */
    .txn-log-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 16px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 0.85rem;
        animation: slideUp 0.3s ease-out;
    }
    .txn-log-item:last-child { border-bottom: none; }
    .txn-dot-ok {
        width: 8px; height: 8px; border-radius: 50%;
        background: #34d399; box-shadow: 0 0 8px rgba(34,197,94,0.5);
        flex-shrink: 0;
    }
    .txn-dot-fraud {
        width: 8px; height: 8px; border-radius: 50%;
        background: #f87171; box-shadow: 0 0 8px rgba(248,113,113,0.5);
        flex-shrink: 0;
    }
    </style>
    """, unsafe_allow_html=True)

    df = load_dataset_sample(n=5000)

    if df.empty:
        st.warning("No data available. Run `python main.py preprocess` first.")
    else:
        atm_tab, stream_tab = st.tabs(["🏧 ATM Simulator", "📡 Live Stream"])

        # ────────────────────────────────────────
        #  TAB 1: ATM MACHINE SIMULATOR
        # ────────────────────────────────────────
        with atm_tab:
            # Initialize session state
            if "atm_history" not in st.session_state:
                st.session_state.atm_history = []

            # Load a model for fraud detection
            _atm_model = None
            _atm_model_name = "N/A"
            for _m_key in ["xgboost", "random_forest", "lightgbm", "naive_bayes"]:
                _smote_path = MODELS_DIR / f"{DATASET_NAME}_smote_{_m_key}.joblib"
                if _smote_path.exists():
                    _atm_model = joblib.load(_smote_path)
                    _atm_model_name = _m_key.replace("_", " ").title()
                    break
                _others = list(MODELS_DIR.glob(f"*_{_m_key}.joblib"))
                if _others:
                    _atm_model = joblib.load(_others[0])
                    _atm_model_name = _m_key.replace("_", " ").title()
                    break

            feature_cols = [c for c in df.columns if c != "target"]

            # ── Layout: ATM on left, controls on right ──
            atm_col, ctrl_col = st.columns([5, 5])

            with ctrl_col:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🏧 Transaction Setup")

                # Card number (simulated)
                card_num = st.text_input(
                    "💳 Card Number",
                    value="4532 •••• •••• 7891",
                    help="Simulated card number",
                    key="atm_card",
                )

                # PIN (simulated)
                pin_input = st.text_input(
                    "🔑 PIN",
                    type="password",
                    max_chars=4,
                    help="Enter any 4-digit PIN",
                    key="atm_pin",
                )

                # Transaction type
                txn_type = st.selectbox(
                    "📋 Transaction Type",
                    ["💵 Cash Withdrawal", "💰 Deposit", "💸 Fund Transfer", "📊 Balance Inquiry"],
                    key="atm_txn_type",
                )

                # Amount
                if txn_type != "📊 Balance Inquiry":
                    amount = st.number_input(
                        "💲 Amount ($)",
                        min_value=1.0,
                        max_value=50000.0,
                        value=250.0,
                        step=50.0,
                        key="atm_amount",
                    )
                else:
                    amount = 0.0

                # Advanced: allow tweaking some features
                with st.expander("⚙️ Advanced — Feature Override"):
                    st.caption("Override V-features (PCA components) to simulate different transaction patterns.")
                    adv_cols = st.columns(3)
                    v_overrides = {}
                    for vi in range(1, 7):
                        with adv_cols[(vi - 1) % 3]:
                            v_overrides[f"V{vi}"] = st.number_input(
                                f"V{vi}", value=0.0, format="%.3f", key=f"atm_v{vi}"
                            )

                process_btn = st.button(
                    "🔄 Process Transaction",
                    type="primary",
                    use_container_width=True,
                    key="atm_process",
                )
                st.markdown('</div>', unsafe_allow_html=True)

            with atm_col:
                # ── ATM Machine Visual ──
                st.markdown("""
                <div class="atm-machine">
                    <div class="atm-brand">
                        <div class="atm-brand-name">FRAUDSHIELD</div>
                        <div class="atm-brand-sub">Secure Banking Terminal</div>
                    </div>
                    <div class="atm-card-slot"></div>
                """, unsafe_allow_html=True)

                atm_screen = st.empty()

                # Default screen
                if not process_btn:
                    atm_screen.markdown("""
                    <div class="atm-screen">
                        <div class="atm-screen-title">FraudShield Secure ATM</div>
                        <div class="atm-screen-main">Welcome</div>
                        <div class="atm-screen-sub">Insert your card and enter PIN to begin</div>
                        <div style="text-align:center; margin-top:18px; font-size:2rem;">🏧</div>
                        <div style="text-align:center; margin-top:8px;">
                            <span style="display:inline-block; width:8px; height:8px; border-radius:50%;
                                         background:#34d399; margin:0 3px; animation:pulseGlow 2s infinite;"></span>
                            <span style="display:inline-block; width:8px; height:8px; border-radius:50%;
                                         background:#818cf8; margin:0 3px;"></span>
                            <span style="display:inline-block; width:8px; height:8px; border-radius:50%;
                                         background:#f472b6; margin:0 3px;"></span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Decorative keypad ──
                st.markdown("""
                    <div class="atm-keypad">
                        <div class="atm-key">1</div><div class="atm-key">2</div><div class="atm-key">3</div>
                        <div class="atm-key">4</div><div class="atm-key">5</div><div class="atm-key">6</div>
                        <div class="atm-key">7</div><div class="atm-key">8</div><div class="atm-key">9</div>
                        <div class="atm-key atm-key-cancel">CANCEL</div>
                        <div class="atm-key">0</div>
                        <div class="atm-key atm-key-enter">ENTER</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Process Transaction ──
            if process_btn:
                if len(pin_input) != 4 or not pin_input.isdigit():
                    atm_screen.markdown("""
                    <div class="atm-screen">
                        <div class="atm-screen-title">Error</div>
                        <div style="text-align:center; font-size:2.5rem; margin:10px 0;">⚠️</div>
                        <div class="atm-screen-main" style="color:#fbbf24; font-size:1.2rem;">
                            INVALID PIN
                        </div>
                        <div class="atm-screen-sub">Please enter a valid 4-digit PIN</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Show processing animation
                    with st.spinner(""):
                        atm_screen.markdown(f"""
                        <div class="atm-screen">
                            <div class="atm-screen-title">Processing</div>
                            <div style="text-align:center; font-size:2rem; margin:12px 0;
                                        animation: shimmer 1s linear infinite;">⏳</div>
                            <div class="atm-screen-main" style="font-size:1.1rem;">
                                Verifying transaction...
                            </div>
                            <div class="atm-screen-sub">
                                {txn_type}<br>Amount: ${amount:,.2f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(1.5)

                    # ── Build feature vector from real data pattern ──
                    rand_idx = np.random.randint(0, len(df))
                    sample_row = df.iloc[rand_idx]
                    input_vals = {f: sample_row[f] for f in feature_cols}
                    input_vals["Amount"] = amount
                    # Apply V-feature overrides
                    for vk, vv in v_overrides.items():
                        if vk in input_vals and vv != 0.0:
                            input_vals[vk] = vv

                    # ── Run fraud detection ──
                    input_df = pd.DataFrame([input_vals])
                    if _atm_model is not None:
                        pred = int(_atm_model.predict(input_df)[0])
                        if hasattr(_atm_model, "predict_proba"):
                            prob = _atm_model.predict_proba(input_df)[0]
                            confidence = prob[1] if pred == 1 else prob[0]
                            fraud_prob = prob[1] * 100
                        else:
                            confidence = None
                            fraud_prob = 100.0 if pred == 1 else 0.0
                    else:
                        # Fallback: use the actual label
                        pred = int(sample_row.get("target", 0))
                        confidence = None
                        fraud_prob = 100.0 if pred == 1 else 0.0

                    txn_id = f"TXN-{int(time.time())}-{np.random.randint(1000, 9999)}"
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    import datetime as _dt

                    # Store in history
                    st.session_state.atm_history.insert(0, {
                        "id": txn_id,
                        "time": timestamp_str,
                        "type": txn_type,
                        "amount": amount,
                        "status": "DECLINED" if pred == 1 else "APPROVED",
                        "fraud_prob": fraud_prob,
                        "card": card_num[-4:] if len(card_num) >= 4 else "****",
                    })

                    # ── Show result on ATM screen ──
                    if pred == 1:
                        # FRAUD — DECLINED
                        conf_str = f"{confidence*100:.1f}%" if confidence else "N/A"
                        atm_screen.markdown(f"""
                        <div class="atm-screen" style="border-color: rgba(239, 68, 68, 0.4);">
                            <div class="atm-screen-title" style="color: rgba(239, 68, 68, 0.8);">
                                ⚠ Security Alert
                            </div>
                            <div style="text-align:center; font-size:3rem; margin:8px 0;
                                        animation: pulseGlow 1.5s ease-in-out infinite;">🚫</div>
                            <div class="atm-screen-main" style="color:#fca5a5; font-size:1.4rem;">
                                TRANSACTION DECLINED
                            </div>
                            <div class="atm-screen-sub" style="color:rgba(252,165,165,0.7);">
                                Suspicious activity detected<br>
                                Fraud probability: {fraud_prob:.1f}%<br>
                                <span style="font-size:0.75rem; margin-top:8px; display:block;">
                                    Please contact your bank immediately
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="atm-declined">
                            <div style="font-size:2rem; margin-bottom:10px;">🚨</div>
                            <div style="font-size:1.4rem; font-weight:800; color:#fca5a5;">
                                FRAUD ALERT — Transaction Blocked
                            </div>
                            <div style="color:rgba(252,165,165,0.7); margin-top:8px;">
                                Model: {_atm_model_name} | Confidence: {conf_str} |
                                Risk Score: {fraud_prob:.1f}%
                            </div>
                            <div style="margin-top:12px;">
                                <span class="status-badge badge-fraud">HIGH RISK</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # LEGITIMATE — APPROVED
                        conf_str = f"{confidence*100:.1f}%" if confidence else "N/A"
                        atm_screen.markdown(f"""
                        <div class="atm-screen" style="border-color: rgba(34, 197, 94, 0.4);">
                            <div class="atm-screen-title" style="color: rgba(34, 197, 94, 0.8);">
                                Transaction Complete
                            </div>
                            <div style="text-align:center; font-size:3rem; margin:8px 0;">✅</div>
                            <div class="atm-screen-main" style="color:#86efac; font-size:1.4rem;">
                                APPROVED
                            </div>
                            <div class="atm-screen-sub" style="color:rgba(134,239,172,0.7);">
                                {txn_type}<br>
                                Amount: ${amount:,.2f}<br>
                                <span style="font-size:0.75rem; margin-top:8px; display:block;">
                                    Please take your card and receipt
                                </span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"""
                        <div class="atm-approved">
                            <div style="font-size:2rem; margin-bottom:10px;">✅</div>
                            <div style="font-size:1.4rem; font-weight:800; color:#86efac;">
                                Transaction Approved
                            </div>
                            <div style="color:rgba(134,239,172,0.7); margin-top:8px;">
                                Model: {_atm_model_name} | Confidence: {conf_str} |
                                Risk Score: {fraud_prob:.1f}%
                            </div>
                            <div style="margin-top:12px;">
                                <span class="status-badge badge-safe">LOW RISK</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Receipt ──
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown("### 🧾 Transaction Receipt")

                    status_label = "DECLINED" if pred == 1 else "APPROVED"
                    status_color = "#dc2626" if pred == 1 else "#16a34a"
                    st.markdown(f"""
                    <div class="atm-receipt">
                        <div class="receipt-header">
                            FRAUDSHIELD BANK<br>
                            <span style="font-size:0.7rem; font-weight:normal;">Secure Banking Terminal</span>
                        </div>
                        <div class="receipt-line"><span>Date:</span><span>{timestamp_str}</span></div>
                        <div class="receipt-line"><span>Txn ID:</span><span>{txn_id}</span></div>
                        <div class="receipt-line"><span>Card:</span><span>****{card_num[-4:] if len(card_num) >= 4 else '****'}</span></div>
                        <div class="receipt-line"><span>Type:</span><span>{txn_type}</span></div>
                        <div class="receipt-line"><span>Amount:</span><span>${amount:,.2f}</span></div>
                        <div style="margin:10px 0; border-top:1px dashed #94a3b8;"></div>
                        <div class="receipt-line">
                            <span style="font-weight:bold;">Status:</span>
                            <span style="font-weight:bold; color:{status_color};">{status_label}</span>
                        </div>
                        <div class="receipt-line"><span>Risk Score:</span><span>{fraud_prob:.1f}%</span></div>
                        <div class="receipt-line"><span>AI Model:</span><span>{_atm_model_name}</span></div>
                        <div class="receipt-footer">
                            Thank you for banking with FraudShield<br>
                            Powered by AI Fraud Detection
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Transaction History ──
            if st.session_state.atm_history:
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown("### 📜 Transaction History")
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)

                for txn in st.session_state.atm_history[:15]:
                    dot_class = "txn-dot-fraud" if txn["status"] == "DECLINED" else "txn-dot-ok"
                    s_badge = "badge-fraud" if txn["status"] == "DECLINED" else "badge-safe"
                    s_label = txn["status"]
                    st.markdown(f"""
                    <div class="txn-log-item">
                        <div class="{dot_class}"></div>
                        <div style="flex:1;">
                            <span style="color:#c0c0e0; font-weight:600;">{txn['id']}</span>
                            <span style="color:rgba(200,200,255,0.4); font-size:0.75rem; margin-left:8px;">
                                {txn['time']}
                            </span>
                        </div>
                        <div style="color:#a0a0c0; font-size:0.85rem;">{txn['type']}</div>
                        <div style="color:#e0e0f0; font-weight:700; min-width:80px; text-align:right;">
                            ${txn['amount']:,.2f}
                        </div>
                        <div><span class="status-badge {s_badge}">{s_label}</span></div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                if st.button("🗑️ Clear History", key="atm_clear"):
                    st.session_state.atm_history = []
                    st.rerun()

        # ────────────────────────────────────────
        #  TAB 2: LIVE STREAM (original)
        # ────────────────────────────────────────
        with stream_tab:
            if st.button("▶️ Start Live Stream", type="primary", use_container_width=True, key="stream_btn"):
                stat_cols = st.columns(4)
                total_ph = stat_cols[0].empty()
                legit_ph = stat_cols[1].empty()
                fraud_ph = stat_cols[2].empty()
                rate_ph = stat_cols[3].empty()

                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                chart_ph = st.empty()
                feed_ph = st.empty()

                np.random.seed(RANDOM_SEED)
                total, frauds, legits = 0, 0, 0
                fraud_times = []
                feed_items = []

                n_stream = min(200, len(df))
                indices = np.random.choice(len(df), n_stream, replace=False)

                for i, idx in enumerate(indices):
                    row = df.iloc[idx]
                    is_fraud = int(row["target"])
                    amount_val = row["Amount"]

                    total += 1
                    if is_fraud:
                        frauds += 1
                    else:
                        legits += 1

                    fraud_times.append({"step": i, "cumulative_fraud": frauds, "total": total,
                                        "fraud_rate": frauds / total * 100})

                    status = "FRAUD" if is_fraud else "OK"
                    badge = "badge-fraud" if is_fraud else "badge-safe"
                    feed_items.insert(0,
                        f'<span class="status-badge {badge}">{status}</span>'
                        f' Txn #{total:04d} — ${amount_val:.2f}')

                    if i % 5 == 0 or i == n_stream - 1:
                        rate = frauds / total * 100

                        total_ph.markdown(f"""<div class="metric-card">
                            <div class="metric-icon">📦</div>
                            <div class="metric-value">{total}</div>
                            <div class="metric-label">Processed</div></div>""",
                            unsafe_allow_html=True)
                        legit_ph.markdown(f"""<div class="metric-card">
                            <div class="metric-icon">✅</div>
                            <div class="metric-value">{legits}</div>
                            <div class="metric-label">Legitimate</div></div>""",
                            unsafe_allow_html=True)
                        fraud_ph.markdown(f"""<div class="metric-card">
                            <div class="metric-icon">🚨</div>
                            <div class="metric-value">{frauds}</div>
                            <div class="metric-label">Fraudulent</div></div>""",
                            unsafe_allow_html=True)
                        rate_ph.markdown(f"""<div class="metric-card">
                            <div class="metric-icon">📊</div>
                            <div class="metric-value">{rate:.2f}%</div>
                            <div class="metric-label">Fraud Rate</div></div>""",
                            unsafe_allow_html=True)

                        ft_df = pd.DataFrame(fraud_times)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=ft_df["step"], y=ft_df["fraud_rate"],
                            mode="lines", name="Fraud Rate %",
                            line=dict(color="#f472b6", width=3),
                            fill="tozeroy",
                            fillcolor="rgba(244,114,182,0.15)",
                        ))
                        fig = styled_plotly(fig, height=300)
                        fig.update_layout(
                            title="Cumulative Fraud Rate",
                            xaxis_title="Transactions", yaxis_title="Fraud Rate %",
                            yaxis=dict(range=[0, max(rate * 3, 1)]),
                        )
                        chart_ph.plotly_chart(fig, use_container_width=True)

                        feed_html = '<div class="glass-card">' + "<br>".join(feed_items[:10]) + "</div>"
                        feed_ph.markdown(feed_html, unsafe_allow_html=True)

                        time.sleep(0.05)

                st.success(f"Stream complete! Processed {total} transactions — {frauds} fraud detected.")
