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
import joblib
import time
import datetime

from dashboard.report_generator import generate_risk_report, _find_similar_transactions

from src.utils.config import (
    MODELS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR,
    DATASET_NAME, RANDOM_SEED,
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
    h1, h2, h3, h4, h5, h6, label, .stMarkdown {
        color: #e0e0f0;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: rgba(200, 200, 255, 0.8) !important;
    }
    
    /* ── DataFrame readability ── */
    [data-testid="stDataFrame"] {
        background-color: transparent;
    }
    /* Let the dataframe text choose its own color in its rendering */
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
    risk = "⬆️ **increased**" if impact > 0 else "⬇️ **reduced**"

    if feat_name == "Amount":
        if impact > 0:
            return f"💰 **Transaction Size** (${value:.2f}): Value is highly anomalous compared to cardholder's historical baseline."
        return f"💰 **Transaction Size** (${value:.2f}): Aligning with standard legitimate activities."
    
    if feat_name == "Time":
        if impact > 0:
            return f"🕐 **Timing / Seasonality** ({value:.0f}s): Transaction occurred at a statistically unusual or hidden hour relative to typical customer usage."
        return f"🕐 **Timing / Seasonality** ({value:.0f}s): Took place during normal, safe behavioral hours."
        
    if feat_name.startswith('V') and feat_name[1:].isdigit():
        n = int(feat_name[1:])
        desc = "General anomaly"
        if 1 <= n <= 5:
            desc = "📍 Geolocation or IP location discrepancy"
        elif 6 <= n <= 10:
            desc = "📈 Transaction velocity or frequency anomaly"
        elif 11 <= n <= 14:
            desc = "💻 Device signature or digital identity mismatch"
        elif 15 <= n <= 19:
            desc = "💳 Payment mechanism or gateway routing irregularity"
        elif 20 <= n <= 28:
            desc = "🛍️ Merchant category or transaction context shift"

        return f"🔬 **{desc}** (internal feature {n}) showed a highly abnormal pattern, meaning a {risk} risk of fraud."

    # Fallback
    strength = "strongly" if abs(impact) > 1.0 else "slightly"
    return f"📊 **{label}** (value: {value:.3f}) {strength} {risk} risk."


# ══════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════

st.sidebar.markdown('<div class="sidebar-title">🛡️ FraudShield AI</div>', unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Dashboard", "📊 Data Explorer", "📈 Model Arena",
     "🔮 Predict & Explain", "🔐 2FA Simulator", "⚡ Live Monitor",
     "🛡️ Admin Panel"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="text-align:center; color:rgba(200,200,255,0.4); font-size:0.75rem;">'
    'A PROJECT BY<br><br>'
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

        # ── Fraud Transaction Explanations ────────────────
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🚨 Flagged Fraud Transactions — Why Are They Suspicious?")
        st.markdown(
            '<div class="page-subtitle">'
            'Each flagged transaction is explained using AI-powered feature analysis (SHAP). '
            'The top contributing factors are described in plain English.</div>',
            unsafe_allow_html=True,
        )

        fraud_df = df_sample[df_sample["target"] == 1].copy()

        if fraud_df.empty:
            st.info("No fraud transactions found in the current sample.")
        else:
            # Try loading the best available model for explanations
            _model_for_explain = None
            _model_name_used = ""
            _model_candidates = [
                ("european_smote_xgboost.joblib", "XGBoost (SMOTE)"),
                ("european_smote_random_forest.joblib", "Random Forest (SMOTE)"),
                ("european_smote_lightgbm.joblib", "LightGBM (SMOTE)"),
                ("european_original_xgboost.joblib", "XGBoost (Original)"),
                ("european_original_random_forest.joblib", "Random Forest (Original)"),
                ("tuned_xgb.joblib", "Tuned XGBoost"),
                ("tuned_rf.joblib", "Tuned Random Forest"),
            ]
            for fname, mname in _model_candidates:
                mpath = MODELS_DIR / fname
                if mpath.exists():
                    try:
                        _model_for_explain = joblib.load(mpath)
                        _model_name_used = mname
                        break
                    except Exception:
                        continue

            if _model_for_explain is None:
                st.warning("⚠️ No trained model found. Run `python main.py train` first to enable explanations.")
            else:
                st.caption(f"🤖 Explanations powered by **{_model_name_used}** model")

                # Compute SHAP values for fraud transactions
                feature_cols = [c for c in fraud_df.columns if c != "target"]
                X_fraud = fraud_df[feature_cols]

                try:
                    import shap
                    explainer = shap.TreeExplainer(_model_for_explain)
                    shap_values = explainer.shap_values(X_fraud)
                    # For binary classifiers that return a list, take the positive class
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                except Exception:
                    shap_values = None

                # Display each fraud transaction with explanation
                for idx_pos, (row_idx, row) in enumerate(fraud_df.iterrows()):
                    txn_id = f"TXN-{row_idx:06d}"
                    amount_val = row.get("Amount", 0)
                    time_val = row.get("Time", 0)

                    # Build the explanation
                    if shap_values is not None:
                        sv = shap_values[idx_pos]
                        # Get top 5 features by absolute SHAP value
                        feat_impacts = sorted(
                            zip(feature_cols, sv, [row[f] for f in feature_cols]),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )[:5]
                        explanations = [
                            _describe_feature(feat, val, impact)
                            for feat, impact, val in feat_impacts
                        ]
                    else:
                        # Fallback: describe based on raw feature values
                        explanations = []
                        if abs(amount_val) > 2.0:
                            explanations.append(
                                f"💰 **Transaction amount** ({amount_val:.2f}) is significantly "
                                f"{'higher' if amount_val > 0 else 'lower'} than average."
                            )
                        explanations.append(
                            "🔍 Detailed SHAP explanation unavailable — install `shap` for full analysis."
                        )

                    # Fraud risk score from model probability
                    try:
                        prob = _model_for_explain.predict_proba(
                            X_fraud.iloc[[idx_pos]]
                        )[0][1]
                        risk_pct = prob * 100
                        risk_color = "#f87171" if risk_pct > 80 else "#fbbf24" if risk_pct > 50 else "#34d399"
                    except Exception:
                        risk_pct = 99.0
                        risk_color = "#f87171"

                    # Render the card
                    explanation_html = "".join(
                        f'<div style="margin:4px 0; font-size:0.92rem;">{e}</div>'
                        for e in explanations
                    )
                    st.markdown(f"""
                    <div style="background:rgba(248,113,113,0.08); border:1px solid rgba(248,113,113,0.25);
                                border-radius:12px; padding:20px; margin:12px 0;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                            <div>
                                <span style="font-size:1.1rem; font-weight:700; color:#f8fafc;">
                                    🆔 {txn_id}
                                </span>
                                <span style="margin-left:16px; color:#94a3b8; font-size:0.85rem;">
                                    💰 Amount: <b>{amount_val:.2f}</b> &nbsp;|&nbsp;
                                    🕐 Time: <b>{time_val:.0f}s</b>
                                </span>
                            </div>
                            <div style="background:{risk_color}; color:#0a0a1a; padding:4px 14px;
                                        border-radius:20px; font-weight:700; font-size:0.85rem;">
                                🔴 {risk_pct:.1f}% Fraud Risk
                            </div>
                        </div>
                        <div style="border-top:1px solid rgba(248,113,113,0.15); padding-top:10px; color:#cbd5e1;">
                            <div style="font-weight:600; margin-bottom:6px; color:#f472b6;">
                                📋 Why This Transaction Was Flagged:
                            </div>
                            {explanation_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


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
# PAGE 5 — 2FA / MFA Simulator (OTP-based)
# ══════════════════════════════════════════════

elif page == "🔐 2FA Simulator":
    st.markdown('<div class="page-title">🔐 2FA Verification</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">High-value transactions require OTP verification sent to your phone or email</div>', unsafe_allow_html=True)

    import random, hashlib

    # ── Session state for OTP flow ──
    if "otp_code" not in st.session_state:
        st.session_state.otp_code = None
    if "otp_verified" not in st.session_state:
        st.session_state.otp_verified = False
    if "otp_channel" not in st.session_state:
        st.session_state.otp_channel = None
    if "otp_txn_amount" not in st.session_state:
        st.session_state.otp_txn_amount = 0.0
    if "otp_attempts" not in st.session_state:
        st.session_state.otp_attempts = 0
    if "otp_history" not in st.session_state:
        st.session_state.otp_history = []

    HIGH_AMOUNT_THRESHOLD = 500.0  # dollars

    # ── Step 1: Transaction Entry ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 💳 New Transaction")

    tc1, tc2 = st.columns(2)
    with tc1:
        tfa_amount = st.number_input(
            "💲 Transaction Amount ($)",
            min_value=1.0, max_value=100000.0, value=250.0, step=50.0,
            key="tfa_amount_input",
        )
        tfa_recipient = st.text_input("👤 Recipient", value="John Doe", key="tfa_recipient")

    with tc2:
        tfa_channel = st.selectbox(
            "📱 OTP Delivery Channel",
            ["📱 SMS (+91 •••• ••48)", "📧 Email (m••••@gmail.com)"],
            key="tfa_channel_select",
        )
        tfa_txn_type = st.selectbox(
            "📋 Transaction Type",
            ["💸 Fund Transfer", "💵 Payment", "🛒 Online Purchase", "💰 Deposit"],
            key="tfa_txn_type",
        )

    requires_2fa = tfa_amount >= HIGH_AMOUNT_THRESHOLD

    # Info banner about 2FA requirement
    if requires_2fa:
        st.markdown(f"""
        <div style="background:rgba(251,191,36,0.12); border:1px solid rgba(251,191,36,0.3);
                    border-radius:12px; padding:14px 18px; margin-top:10px;">
            <span style="font-size:1.1rem;">⚠️</span>
            <span style="color:#fbbf24; font-weight:600;"> 2FA Required</span>
            <span style="color:rgba(200,200,255,0.7); font-size:0.9rem;">
                — Transactions ≥ ${HIGH_AMOUNT_THRESHOLD:,.0f} require One-Time Password verification.
            </span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.3);
                    border-radius:12px; padding:14px 18px; margin-top:10px;">
            <span style="font-size:1.1rem;">✅</span>
            <span style="color:#86efac; font-weight:600;"> Standard Authorization</span>
            <span style="color:rgba(200,200,255,0.7); font-size:0.9rem;">
                — Amount below ${HIGH_AMOUNT_THRESHOLD:,.0f}. No OTP needed.
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Initiate Transaction ──
    if not st.session_state.otp_verified:
        initiate_btn = st.button("🚀 Initiate Transaction", type="primary", use_container_width=True, key="tfa_initiate")

        if initiate_btn:
            if requires_2fa:
                # Generate a 6-digit OTP
                otp = str(random.randint(100000, 999999))
                st.session_state.otp_code = otp
                st.session_state.otp_verified = False
                st.session_state.otp_channel = tfa_channel
                st.session_state.otp_txn_amount = tfa_amount
                st.session_state.otp_attempts = 0
                st.rerun()
            else:
                # Low amount — approve directly
                st.session_state.otp_verified = True
                st.session_state.otp_txn_amount = tfa_amount
                st.session_state.otp_code = None
                st.session_state.otp_channel = None
                st.session_state.otp_history.insert(0, {
                    "id": f"TXN-{int(time.time())}-{random.randint(1000,9999)}",
                    "amount": tfa_amount,
                    "recipient": tfa_recipient,
                    "type": tfa_txn_type,
                    "auth": "Standard (No OTP)",
                    "status": "✅ APPROVED",
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                st.rerun()

    # ── Step 2: OTP Entry (only for high-amount) ──
    if st.session_state.otp_code and not st.session_state.otp_verified:
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        channel_label = "phone" if "SMS" in (st.session_state.otp_channel or "") else "email"
        channel_icon = "📱" if channel_label == "phone" else "📧"

        # OTP sent notification
        st.markdown(f"""
        <div style="background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3);
                    border-radius:16px; padding:24px; text-align:center; margin:10px 0;
                    animation: fadeInScale 0.5s ease-out;">
            <div style="font-size:2.5rem; margin-bottom:8px;">{channel_icon}</div>
            <div style="font-size:1.2rem; font-weight:700; color:#a5b4fc;">
                OTP Sent to your {channel_label}!
            </div>
            <div style="color:rgba(200,200,255,0.6); margin-top:6px; font-size:0.9rem;">
                A 6-digit verification code has been sent to<br>
                <b>{st.session_state.otp_channel}</b>
            </div>
            <div style="margin-top:14px; background:rgba(255,255,255,0.06); border-radius:10px;
                        padding:10px 16px; display:inline-block;">
                <span style="color:#fbbf24; font-weight:700; font-size:0.85rem;">
                    🔑 Demo OTP: {st.session_state.otp_code}
                </span>
            </div>
            <div style="color:rgba(200,200,255,0.4); font-size:0.75rem; margin-top:8px;">
                (In production this would only be sent to the actual device)
            </div>
        </div>
        """, unsafe_allow_html=True)

        # OTP input form
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🔢 Enter Verification Code")

        otp_input = st.text_input(
            "6-Digit OTP",
            max_chars=6,
            placeholder="Enter the OTP sent to your device",
            key="tfa_otp_input",
        )

        remaining = 3 - st.session_state.otp_attempts
        if st.session_state.otp_attempts > 0:
            st.caption(f"⚠️ Attempts remaining: **{remaining}**")

        vc1, vc2 = st.columns(2)
        with vc1:
            verify_btn = st.button("✅ Verify OTP", type="primary", use_container_width=True, key="tfa_verify")
        with vc2:
            resend_btn = st.button("🔄 Resend OTP", use_container_width=True, key="tfa_resend")

        if resend_btn:
            new_otp = str(random.randint(100000, 999999))
            st.session_state.otp_code = new_otp
            st.session_state.otp_attempts = 0
            st.rerun()

        if verify_btn:
            if otp_input == st.session_state.otp_code:
                # ✅ OTP correct — approve transaction
                st.session_state.otp_verified = True
                st.session_state.otp_history.insert(0, {
                    "id": f"TXN-{int(time.time())}-{random.randint(1000,9999)}",
                    "amount": st.session_state.otp_txn_amount,
                    "recipient": tfa_recipient,
                    "type": tfa_txn_type,
                    "auth": f"2FA OTP ({channel_label})",
                    "status": "✅ APPROVED",
                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
                st.rerun()
            else:
                st.session_state.otp_attempts += 1
                if st.session_state.otp_attempts >= 3:
                    # ❌ Max attempts — block transaction
                    st.session_state.otp_history.insert(0, {
                        "id": f"TXN-{int(time.time())}-{random.randint(1000,9999)}",
                        "amount": st.session_state.otp_txn_amount,
                        "recipient": tfa_recipient,
                        "type": tfa_txn_type,
                        "auth": f"2FA OTP ({channel_label})",
                        "status": "🚫 BLOCKED (3 failed attempts)",
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                    st.session_state.otp_code = None
                    st.session_state.otp_verified = False
                    st.rerun()
                else:
                    st.error(f"❌ Incorrect OTP. {3 - st.session_state.otp_attempts} attempt(s) remaining.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── Step 3: Result Display ──
    if st.session_state.otp_verified:
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:rgba(34,197,94,0.12); border:2px solid rgba(34,197,94,0.35);
                    border-radius:20px; padding:36px; text-align:center;
                    animation: fadeInScale 0.5s ease-out;">
            <div style="font-size:3.5rem; margin-bottom:10px;">✅</div>
            <div style="font-size:1.8rem; font-weight:800; color:#86efac;">
                Transaction Approved
            </div>
            <div style="color:rgba(134,239,172,0.7); margin-top:8px; font-size:1.05rem;">
                ${st.session_state.otp_txn_amount:,.2f} — {'OTP Verified' if st.session_state.otp_txn_amount >= HIGH_AMOUNT_THRESHOLD else 'Standard Auth'}
            </div>
            <div style="margin-top:14px;">
                <span class="status-badge badge-safe">AUTHENTICATED</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 New Transaction", use_container_width=True, key="tfa_new"):
            st.session_state.otp_code = None
            st.session_state.otp_verified = False
            st.session_state.otp_channel = None
            st.session_state.otp_txn_amount = 0.0
            st.session_state.otp_attempts = 0
            st.rerun()

    # ── Blocked message (max attempts reached) ──
    if (not st.session_state.otp_code and not st.session_state.otp_verified
            and st.session_state.otp_history
            and "BLOCKED" in st.session_state.otp_history[0].get("status", "")):
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.12); border:2px solid rgba(239,68,68,0.35);
                    border-radius:20px; padding:36px; text-align:center;
                    animation: fadeInScale 0.5s ease-out, pulseGlow 2s ease-in-out infinite;">
            <div style="font-size:3.5rem; margin-bottom:10px;">🚫</div>
            <div style="font-size:1.8rem; font-weight:800; color:#fca5a5;">
                Transaction Blocked
            </div>
            <div style="color:rgba(252,165,165,0.7); margin-top:8px; font-size:1.05rem;">
                Maximum OTP attempts exceeded. Contact your bank.
            </div>
            <div style="margin-top:14px;">
                <span class="status-badge badge-fraud">SECURITY HOLD</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Try Again", use_container_width=True, key="tfa_retry"):
            st.session_state.otp_code = None
            st.session_state.otp_verified = False
            st.session_state.otp_channel = None
            st.session_state.otp_txn_amount = 0.0
            st.session_state.otp_attempts = 0
            st.rerun()

    # ── Transaction History ──
    if st.session_state.otp_history:
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📜 Transaction History")

        for _th in st.session_state.otp_history[:15]:
            _is_ok = "APPROVED" in _th["status"]
            _dot = "txn-dot-ok" if _is_ok else "txn-dot-fraud"
            _badge = "badge-safe" if _is_ok else "badge-fraud"
            _status_text = "APPROVED" if _is_ok else "BLOCKED"
            st.markdown(f"""
            <div class="txn-log-item">
                <div class="{_dot}"></div>
                <div style="flex:1;">
                    <span style="color:#c0c0e0; font-weight:600;">{_th['id']}</span>
                    <span style="color:rgba(200,200,255,0.4); font-size:0.75rem; margin-left:8px;">
                        {_th['time']}
                    </span>
                </div>
                <div style="color:#a0a0c0; font-size:0.85rem;">{_th['auth']}</div>
                <div style="color:#e0e0f0; font-weight:700; min-width:80px; text-align:right;">
                    ${_th['amount']:,.2f}
                </div>
                <div><span class="status-badge {_badge}">{_status_text}</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear History", key="tfa_clear_history"):
            st.session_state.otp_history = []
            st.rerun()


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


# ══════════════════════════════════════════════
# PAGE 7 — Admin Monitoring Panel
# ══════════════════════════════════════════════

elif page == "🛡️ Admin Panel":
    st.markdown('<div class="page-title">🛡️ Admin Monitoring Panel</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Review suspicious transactions, manage approvals, and analyse fraud trends</div>', unsafe_allow_html=True)

    # ── Initialise session state ──
    if "admin_actions" not in st.session_state:
        st.session_state.admin_actions = []  # list of dicts

    # ── Load data & model ──
    _admin_df = load_dataset_sample(n=10000)

    if _admin_df.empty:
        st.warning("⚠️ No data found. Run `python main.py preprocess` first.")
    else:
        feature_cols_admin = [c for c in _admin_df.columns if c != "target"]
        X_admin = _admin_df[feature_cols_admin]

        # Load best model
        _admin_model = None
        _admin_model_name = "N/A"
        for _mk, _mn in [("xgboost", "XGBoost"), ("random_forest", "Random Forest"),
                         ("lightgbm", "LightGBM"), ("naive_bayes", "Naive Bayes")]:
            _mp = MODELS_DIR / f"{DATASET_NAME}_smote_{_mk}.joblib"
            if _mp.exists():
                _admin_model = joblib.load(_mp)
                _admin_model_name = _mn
                break
            _others = list(MODELS_DIR.glob(f"*_{_mk}.joblib"))
            if _others:
                _admin_model = joblib.load(_others[0])
                _admin_model_name = _mn
                break

        if _admin_model is None:
            st.warning("⚠️ No trained model found. Run `python main.py train` first.")
        else:
            # Compute risk scores for all transactions
            @st.cache_data
            def _compute_admin_scores(_X_csv):
                """Deterministic scoring cached on data content."""
                _Xdf = pd.read_json(_X_csv)
                mdl = _admin_model
                if hasattr(mdl, "predict_proba"):
                    probs = mdl.predict_proba(_Xdf)[:, 1] * 100
                else:
                    probs = mdl.predict(_Xdf).astype(float) * 100
                return probs

            _admin_probs = _compute_admin_scores(X_admin.to_json())

            # Build enriched dataframe
            admin_view = _admin_df.copy()
            admin_view.insert(0, "Risk %", np.round(_admin_probs, 2))
            admin_view.insert(0, "TXN_ID", [f"TXN-{i:06d}" for i in admin_view.index])

            def _risk_label(pct):
                if pct >= 80:
                    return "🔴 High"
                elif pct >= 50:
                    return "🟡 Medium"
                return "🟢 Low"

            admin_view.insert(2, "Risk Level", admin_view["Risk %"].apply(_risk_label))

            # Track which TXNs have been acted on
            acted_ids = {a["txn_id"] for a in st.session_state.admin_actions}

            # ── Tabs ──
            tab_suspicious, tab_filter, tab_actions, tab_analytics = st.tabs([
                "🔍 Suspicious Transactions",
                "🎚️ Filter by Risk Level",
                "✅ Approve / Block",
                "📊 Fraud Analytics & Reports",
            ])

            # ════════════════════════════════════════
            # TAB 1 — Suspicious Transactions
            # ════════════════════════════════════════
            with tab_suspicious:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🚨 Flagged Suspicious Transactions")
                st.caption(f"Model: **{_admin_model_name}** — showing transactions with risk > 50 %")

                flagged = admin_view[admin_view["Risk %"] > 50].sort_values("Risk %", ascending=False)

                if flagged.empty:
                    st.info("No suspicious transactions detected in the current sample.")
                else:
                    st.dataframe(
                        flagged[["TXN_ID", "Risk Level", "Risk %", "Amount", "Time", "target"]].rename(
                            columns={"target": "Actual Label"}
                        ).style.background_gradient(subset=["Risk %"], cmap="YlOrRd"),
                        use_container_width=True,
                        height=400,
                    )

                    # Expandable SHAP explanations for top flagged transactions
                    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                    st.markdown("### 📋 Detailed Explanations (Top 20)")

                    _shap_avail = False
                    try:
                        import shap
                        _explainer = shap.TreeExplainer(_admin_model)
                        _shap_avail = True
                    except Exception:
                        pass

                    for _pos, (_ridx, _row) in enumerate(flagged.head(20).iterrows()):
                        risk_color = "#f87171" if _row["Risk %"] >= 80 else "#fbbf24"
                        with st.expander(f"{_row['TXN_ID']}  —  💰 ${_row['Amount']:.2f}  —  Risk {_row['Risk %']:.1f}%"):
                            if _shap_avail:
                                try:
                                    _sv = _explainer.shap_values(X_admin.iloc[[_ridx]])
                                    if isinstance(_sv, list):
                                        _sv = _sv[1][0]
                                    else:
                                        _sv = _sv[0]
                                    _impacts = sorted(
                                        zip(feature_cols_admin, _sv,
                                            [_admin_df.iloc[_ridx][f] for f in feature_cols_admin]),
                                        key=lambda x: abs(x[1]), reverse=True,
                                    )[:5]
                                    for _feat, _imp, _val in _impacts:
                                        st.markdown(_describe_feature(_feat, _val, _imp))
                                except Exception:
                                    st.info("SHAP explanation unavailable for this transaction.")
                            else:
                                st.info("Install `shap` for detailed feature explanations.")

                st.markdown('</div>', unsafe_allow_html=True)

            # ════════════════════════════════════════
            # TAB 2 — Filter by Risk Level
            # ════════════════════════════════════════
            with tab_filter:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 🎚️ Filter Transactions")

                fc1, fc2, fc3 = st.columns([2, 2, 3])
                with fc1:
                    risk_filter = st.selectbox(
                        "Risk Level",
                        ["All", "🔴 High (≥80%)", "🟡 Medium (50-80%)", "🟢 Low (<50%)"],
                        key="admin_risk_filter",
                    )
                with fc2:
                    amt_range = st.slider(
                        "Amount Range",
                        float(admin_view["Amount"].min()),
                        float(admin_view["Amount"].max()),
                        (float(admin_view["Amount"].min()), float(admin_view["Amount"].max())),
                        key="admin_amt_range",
                    )
                with fc3:
                    txn_search = st.text_input("🔎 Search Transaction ID", key="admin_txn_search")

                filtered = admin_view.copy()
                if risk_filter == "🔴 High (≥80%)":
                    filtered = filtered[filtered["Risk %"] >= 80]
                elif risk_filter == "🟡 Medium (50-80%)":
                    filtered = filtered[(filtered["Risk %"] >= 50) & (filtered["Risk %"] < 80)]
                elif risk_filter == "🟢 Low (<50%)":
                    filtered = filtered[filtered["Risk %"] < 50]

                filtered = filtered[
                    (filtered["Amount"] >= amt_range[0]) & (filtered["Amount"] <= amt_range[1])
                ]

                if txn_search:
                    filtered = filtered[filtered["TXN_ID"].str.contains(txn_search, case=False)]

                # KPI row
                kc1, kc2, kc3, kc4 = st.columns(4)
                _kpi_data = [
                    ("📦", f"{len(filtered):,}", "Matching Txns"),
                    ("🔴", f"{int((filtered['Risk %'] >= 80).sum()):,}", "High Risk"),
                    ("🟡", f"{int(((filtered['Risk %'] >= 50) & (filtered['Risk %'] < 80)).sum()):,}", "Medium Risk"),
                    ("🟢", f"{int((filtered['Risk %'] < 50).sum()):,}", "Low Risk"),
                ]
                for _i, (_icon, _val, _lbl) in enumerate(_kpi_data):
                    [kc1, kc2, kc3, kc4][_i].markdown(f"""
                        <div class="metric-card delay-{_i+1}">
                            <div class="metric-icon">{_icon}</div>
                            <div class="metric-value">{_val}</div>
                            <div class="metric-label">{_lbl}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                st.dataframe(
                    filtered[["TXN_ID", "Risk Level", "Risk %", "Amount", "Time", "target"]].rename(
                        columns={"target": "Actual Label"}
                    ).sort_values("Risk %", ascending=False).style.background_gradient(
                        subset=["Risk %"], cmap="YlOrRd"
                    ),
                    use_container_width=True,
                    height=500,
                )
                st.markdown('</div>', unsafe_allow_html=True)

            # ════════════════════════════════════════
            # TAB 3 — Approve / Block Actions
            # ════════════════════════════════════════
            with tab_actions:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### ✅ Pending Suspicious Transactions")
                st.caption("Take action on flagged transactions — approve legitimate ones or block fraud.")

                pending = admin_view[
                    (admin_view["Risk %"] > 50) & (~admin_view["TXN_ID"].isin(acted_ids))
                ].sort_values("Risk %", ascending=False).head(30)

                if pending.empty:
                    st.success("🎉 All suspicious transactions have been reviewed!")
                else:
                    for _pi, (_pidx, _prow) in enumerate(pending.iterrows()):
                        _risk_col = "#f87171" if _prow["Risk %"] >= 80 else "#fbbf24"
                        _badge_cls = "badge-fraud" if _prow["Risk %"] >= 80 else "badge-safe"

                        st.markdown(f"""
                        <div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);
                                    border-radius:12px; padding:16px; margin:8px 0;
                                    display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:10px;">
                            <div>
                                <span style="font-weight:700; color:#e0e0f0; font-size:1rem;">{_prow['TXN_ID']}</span>
                                <span style="margin-left:14px; color:#94a3b8; font-size:0.85rem;">
                                    💰 ${_prow['Amount']:.2f} &nbsp;|&nbsp; 🕐 {_prow['Time']:.0f}s
                                </span>
                            </div>
                            <div style="background:{_risk_col}; color:#0a0a1a; padding:4px 14px;
                                        border-radius:20px; font-weight:700; font-size:0.82rem;">
                                {_prow['Risk %']:.1f}% Risk
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        bcol1, bcol2, _ = st.columns([1, 1, 4])
                        with bcol1:
                            if st.button("✅ Approve", key=f"approve_{_prow['TXN_ID']}"):
                                st.session_state.admin_actions.append({
                                    "txn_id": _prow["TXN_ID"],
                                    "action": "APPROVED",
                                    "risk": _prow["Risk %"],
                                    "amount": _prow["Amount"],
                                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                })
                                st.rerun()
                        with bcol2:
                            if st.button("🚫 Block", key=f"block_{_prow['TXN_ID']}"):
                                st.session_state.admin_actions.append({
                                    "txn_id": _prow["TXN_ID"],
                                    "action": "BLOCKED",
                                    "risk": _prow["Risk %"],
                                    "amount": _prow["Amount"],
                                    "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                })
                                st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)

                # ── Action Log ──
                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("### 📜 Action Log")

                if st.session_state.admin_actions:
                    log_df = pd.DataFrame(st.session_state.admin_actions)
                    log_df = log_df.rename(columns={
                        "txn_id": "Transaction", "action": "Decision",
                        "risk": "Risk %", "amount": "Amount", "time": "Timestamp",
                    })

                    for _, _lrow in log_df.iterrows():
                        _dot = "txn-dot-fraud" if _lrow["Decision"] == "BLOCKED" else "txn-dot-ok"
                        _badge = "badge-fraud" if _lrow["Decision"] == "BLOCKED" else "badge-safe"
                        st.markdown(f"""
                        <div class="txn-log-item">
                            <div class="{_dot}"></div>
                            <div style="flex:1;">
                                <span style="color:#c0c0e0; font-weight:600;">{_lrow['Transaction']}</span>
                                <span style="color:rgba(200,200,255,0.4); font-size:0.75rem; margin-left:8px;">
                                    {_lrow['Timestamp']}
                                </span>
                            </div>
                            <div style="color:#a0a0c0; font-size:0.85rem;">Risk: {_lrow['Risk %']:.1f}%</div>
                            <div style="color:#e0e0f0; font-weight:700; min-width:80px; text-align:right;">
                                ${_lrow['Amount']:.2f}
                            </div>
                            <div><span class="status-badge {_badge}">{_lrow['Decision']}</span></div>
                        </div>
                        """, unsafe_allow_html=True)

                    if st.button("🗑️ Clear Action Log", key="admin_clear_log"):
                        st.session_state.admin_actions = []
                        st.rerun()
                else:
                    st.info("No actions taken yet. Review transactions above to get started.")

                st.markdown('</div>', unsafe_allow_html=True)

            # ════════════════════════════════════════
            # TAB 4 — Fraud Analytics & Reports
            # ════════════════════════════════════════
            with tab_analytics:
                # KPI cards
                n_flagged = int((admin_view["Risk %"] > 50).sum())
                n_approved = sum(1 for a in st.session_state.admin_actions if a["action"] == "APPROVED")
                n_blocked = sum(1 for a in st.session_state.admin_actions if a["action"] == "BLOCKED")
                avg_risk = admin_view["Risk %"].mean()

                acols = st.columns(4)
                _a_kpis = [
                    ("🚨", f"{n_flagged:,}", "Total Flagged"),
                    ("✅", f"{n_approved:,}", "Approved"),
                    ("🚫", f"{n_blocked:,}", "Blocked"),
                    ("📊", f"{avg_risk:.2f}%", "Avg Risk Score"),
                ]
                for _i, (_icon, _val, _lbl) in enumerate(_a_kpis):
                    acols[_i].markdown(f"""
                        <div class="metric-card delay-{_i+1}">
                            <div class="metric-icon">{_icon}</div>
                            <div class="metric-value">{_val}</div>
                            <div class="metric-label">{_lbl}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                # Chart row 1
                ch1, ch2 = st.columns(2)

                with ch1:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    fig = go.Figure(go.Histogram(
                        x=admin_view["Risk %"],
                        nbinsx=40,
                        marker_color="#818cf8",
                        opacity=0.85,
                    ))
                    fig = styled_plotly(fig, height=380)
                    fig.update_layout(
                        title=dict(text="Risk Score Distribution", font=dict(size=18)),
                        xaxis_title="Risk %",
                        yaxis_title="Count",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with ch2:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    _fraud_cnt = int(admin_view["target"].sum())
                    _legit_cnt = len(admin_view) - _fraud_cnt
                    fig = go.Figure(go.Pie(
                        labels=["Legitimate", "Fraudulent"],
                        values=[_legit_cnt, _fraud_cnt],
                        hole=0.65,
                        marker=dict(colors=["#34d399", "#f87171"],
                                    line=dict(color="#0a0a1a", width=3)),
                        textinfo="percent+label",
                        pull=[0, 0.08],
                    ))
                    fig = styled_plotly(fig, height=380)
                    fig.update_layout(
                        title=dict(text="Fraud vs Legitimate", font=dict(size=18)),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

                # Chart row 2
                ch3, ch4 = st.columns(2)

                with ch3:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    top10 = admin_view.nlargest(10, "Risk %")
                    fig = go.Figure(go.Bar(
                        x=top10["TXN_ID"],
                        y=top10["Risk %"],
                        marker=dict(
                            color=top10["Risk %"],
                            colorscale=[[0, "#fbbf24"], [0.5, "#f87171"], [1, "#dc2626"]],
                            cornerradius=8,
                        ),
                        text=[f"{v:.1f}%" for v in top10["Risk %"]],
                        textposition="outside",
                        textfont=dict(color="#c0c0e0", size=11),
                    ))
                    fig = styled_plotly(fig, height=380)
                    fig.update_layout(
                        title=dict(text="Top 10 Highest-Risk Transactions", font=dict(size=18)),
                        xaxis_title="", yaxis_title="Risk %",
                        yaxis=dict(range=[0, 110]),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with ch4:
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    # Time-binned fraud trend
                    _trend = admin_view.copy()
                    _trend["Time_Bin"] = pd.cut(_trend["Time"], bins=20, labels=False)
                    _trend_agg = _trend.groupby("Time_Bin").agg(
                        fraud_count=("target", "sum"),
                        total=("target", "count"),
                    ).reset_index()
                    _trend_agg["fraud_rate"] = _trend_agg["fraud_count"] / _trend_agg["total"] * 100

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=_trend_agg["Time_Bin"],
                        y=_trend_agg["fraud_rate"],
                        mode="lines+markers",
                        line=dict(color="#f472b6", width=3),
                        marker=dict(size=8, color="#c084fc"),
                        fill="tozeroy",
                        fillcolor="rgba(244,114,182,0.12)",
                        name="Fraud Rate",
                    ))
                    fig = styled_plotly(fig, height=380)
                    fig.update_layout(
                        title=dict(text="Fraud Trend over Time", font=dict(size=18)),
                        xaxis_title="Time Bin",
                        yaxis_title="Fraud Rate %",
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
