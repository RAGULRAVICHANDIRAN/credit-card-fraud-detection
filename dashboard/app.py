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
    'Built with ❤️ by FraudShield AI<br>v2.0 — Animated Edition</div>',
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

    model_files = list(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        st.warning("No saved models found.")
    else:
        model_choice = st.selectbox("🤖 Select Model", [f.stem for f in model_files])
        model_path = MODELS_DIR / f"{model_choice}.joblib"
        model = joblib.load(model_path)

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
# PAGE 6 — Live Monitor
# ══════════════════════════════════════════════

elif page == "⚡ Live Monitor":
    st.markdown('<div class="page-title">⚡ Live Fraud Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Simulated real-time transaction stream with fraud detection</div>', unsafe_allow_html=True)

    df = load_dataset_sample(n=5000)

    if df.empty:
        st.warning("No data available.")
    else:
        if st.button("▶️ Start Live Stream", type="primary", use_container_width=True):
            # Stats containers
            stat_cols = st.columns(4)
            total_ph = stat_cols[0].empty()
            legit_ph = stat_cols[1].empty()
            fraud_ph = stat_cols[2].empty()
            rate_ph = stat_cols[3].empty()

            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

            chart_ph = st.empty()
            feed_ph = st.empty()

            # Simulate stream
            np.random.seed(RANDOM_SEED)
            total, frauds, legits = 0, 0, 0
            fraud_times = []
            feed_items = []

            n_stream = min(200, len(df))
            indices = np.random.choice(len(df), n_stream, replace=False)

            for i, idx in enumerate(indices):
                row = df.iloc[idx]
                is_fraud = int(row["target"])
                amount = row["Amount"]

                total += 1
                if is_fraud:
                    frauds += 1
                else:
                    legits += 1

                fraud_times.append({"step": i, "cumulative_fraud": frauds, "total": total,
                                    "fraud_rate": frauds/total*100})

                # Feed
                status = "🚨 FRAUD" if is_fraud else "✅ OK"
                badge = "badge-fraud" if is_fraud else "badge-safe"
                feed_items.insert(0, f'<span class="status-badge {badge}">{status}</span>'
                                  f' Txn #{total:04d} — Amount: {amount:.2f}')

                # Update every 5 txns for speed
                if i % 5 == 0 or i == n_stream - 1:
                    rate = frauds/total*100

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

                    # Cumulative chart
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

                    # Feed (last 10)
                    feed_html = '<div class="glass-card">' + "<br>".join(feed_items[:10]) + "</div>"
                    feed_ph.markdown(feed_html, unsafe_allow_html=True)

                    time.sleep(0.05)

            st.success(f"✅ Stream complete! Processed {total} transactions — {frauds} fraud detected.")
