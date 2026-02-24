"""
Streamlit Dashboard for Credit Card Fraud Detection Project.

Launch:  streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

from src.utils.config import (
    MODELS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR,
    DS_EUROPEAN, DS_SPARKOV, ALL_STRATEGIES, ALL_MODELS,
    RANDOM_SEED,
)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700; color: #1E3A5F;}
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
    }
    .metric-value {font-size: 2rem; font-weight: 700;}
    .metric-label {font-size: 0.85rem; opacity: 0.85;}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════

st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "📈 Model Performance", "🔮 Prediction Interface",
     "🔍 Explainability", "🔐 2FA Simulation", "📅 Time Series"],
)


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

# Feature name → human-friendly label
_FEATURE_LABELS = {
    "Amount": "Transaction amount",
    "amt": "Transaction amount",
    "Time": "Transaction time",
    "trans_hour": "Hour of transaction",
    "trans_day_of_week": "Day of the week",
    "trans_month": "Month of transaction",
    "category_encoded": "Merchant category",
    "city_pop": "City population",
    "lat": "Latitude",
    "long": "Longitude",
    "merch_lat": "Merchant latitude",
    "merch_long": "Merchant longitude",
    "age": "Cardholder age",
    "job_encoded": "Cardholder occupation",
}


def _describe_feature(feat_name: str, value: float, impact: float) -> str:
    """Turn a feature name, its value, and its SHAP/z-score into plain English."""
    label = _FEATURE_LABELS.get(feat_name, feat_name)

    # Direction of influence
    if impact > 0:
        direction_word = "increases"
        risk_word = "⬆️ higher"
    else:
        direction_word = "decreases"
        risk_word = "⬇️ lower"

    # Special readable descriptions for known features
    if feat_name in ("Amount", "amt"):
        if impact > 0:
            return (f"**{label}** (${value:.2f}) is unusually high "
                    "— large transactions are a common fraud signal.")
        else:
            return (f"**{label}** (${value:.2f}) is within a normal range, "
                    "which is typical of legitimate transactions.")

    if feat_name in ("trans_hour", "Time"):
        if impact > 0:
            return (f"**{label}** ({value:.0f}) falls in an unusual time window "
                    "— fraud often occurs during off-peak hours (late night/early morning).")
        else:
            return (f"**{label}** ({value:.0f}) is during normal business hours, "
                    "which is typical of legitimate transactions.")

    if feat_name == "city_pop":
        if impact > 0:
            return (f"**{label}** ({value:,.0f}) is atypical for this type of transaction "
                    "— fraud can cluster in certain population zones.")
        else:
            return (f"**{label}** ({value:,.0f}) is consistent with normal transaction patterns.")

    # PCA features (V1–V28 in European dataset)
    if feat_name.startswith("V") and feat_name[1:].isdigit():
        feat_num = feat_name[1:]
        if impact > 0:
            return (f"**Hidden pattern {feat_num}** (value: {value:.3f}) shows an anomalous "
                    f"pattern {risk_word} fraud risk — this feature captures spending behaviour "
                    "deviations detected by the model.")
        else:
            return (f"**Hidden pattern {feat_num}** (value: {value:.3f}) is within the normal "
                    f"range, {risk_word} fraud risk.")

    # Generic fallback
    strength = "strongly" if abs(impact) > 1.0 else "slightly"
    return (f"**{label}** (value: {value:.3f}) {strength} {direction_word} "
            f"fraud risk ({risk_word}).")

@st.cache_data
def load_results():
    path = MODELS_DIR / "existing_results.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_dataset_sample(ds_name, n=5000):
    try:
        X = pd.read_csv(PROCESSED_DATA_DIR / ds_name / "X_test.csv")
        y = pd.read_csv(PROCESSED_DATA_DIR / ds_name / "y_test.csv").squeeze()
        df = X.copy()
        df["target"] = y.values
        return df.head(n)
    except FileNotFoundError:
        return pd.DataFrame()


# ══════════════════════════════════════════════
# PAGE 1 – Data Overview
# ══════════════════════════════════════════════

if page == "📊 Data Overview":
    st.markdown('<p class="main-header">📊  Data Overview</p>', unsafe_allow_html=True)

    ds = st.selectbox("Select Dataset", [DS_EUROPEAN, DS_SPARKOV])
    df = load_dataset_sample(ds)

    if df.empty:
        st.warning("Processed data not found. Run `python main.py preprocess` first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        n_total = len(df)
        n_fraud = int(df["target"].sum())
        col1.metric("Samples (shown)", f"{n_total:,}")
        col2.metric("Fraudulent", f"{n_fraud:,}")
        col3.metric("Legitimate", f"{n_total - n_fraud:,}")
        col4.metric("Fraud %", f"{n_fraud / n_total * 100:.2f}%")

        st.subheader("Class Distribution")
        fig = px.histogram(df, x="target", color="target",
                           labels={"target": "Class"},
                           color_discrete_map={0: "#55A868", 1: "#C44E52"},
                           title="Class Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Statistics")
        st.dataframe(df.describe().T.style.format("{:.3f}"), use_container_width=True)

        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include=[np.number]).corr()
        fig_corr = px.imshow(corr, color_continuous_scale="RdBu_r",
                             title="Feature Correlations", aspect="auto")
        st.plotly_chart(fig_corr, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 2 – Model Performance
# ══════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.markdown('<p class="main-header">📈  Model Performance</p>', unsafe_allow_html=True)

    df = load_results()
    if df.empty:
        st.warning("No results found. Run training first.")
    else:
        st.subheader("Results Table")
        st.dataframe(df.style.format({
            "accuracy": "{:.4f}", "precision": "{:.4f}",
            "recall": "{:.4f}", "f1": "{:.4f}", "roc_auc": "{:.4f}",
        }), use_container_width=True)

        metric = st.selectbox("Metric to visualise", ["f1", "accuracy", "precision", "recall", "roc_auc"])

        fig = px.bar(
            df, x="model", y=metric, color="strategy",
            barmode="group", facet_col="dataset",
            title=f"{metric.upper()} by Model, Strategy, and Dataset",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Heatmap View")
        pivot = df.pivot_table(values=metric, index=["dataset", "strategy"],
                               columns="model")
        fig_hm = px.imshow(pivot, text_auto=".3f",
                           color_continuous_scale="Viridis",
                           title=f"{metric.upper()} Heatmap")
        st.plotly_chart(fig_hm, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 3 – Prediction Interface
# ══════════════════════════════════════════════

elif page == "🔮 Prediction Interface":
    st.markdown('<p class="main-header">🔮  Predict Fraud</p>', unsafe_allow_html=True)

    st.info("Enter transaction features below to get a fraud prediction.")

    model_files = list(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        st.warning("No saved models found.")
    else:
        model_choice = st.selectbox("Model", [f.stem for f in model_files])
        model_path = MODELS_DIR / f"{model_choice}.joblib"
        model = joblib.load(model_path)

        # Determine expected features
        ds_name = model_choice.split("_")[0]
        sample_df = load_dataset_sample(ds_name, n=1)

        if not sample_df.empty:
            feature_cols = [c for c in sample_df.columns if c != "target"]
            st.write(f"**Features expected**: {len(feature_cols)}")

            cols = st.columns(4)
            values = {}
            for i, feat in enumerate(feature_cols):
                with cols[i % 4]:
                    values[feat] = st.number_input(feat, value=0.0, format="%.4f",
                                                   key=f"feat_{feat}")

            if st.button("🔍 Predict", type="primary"):
                input_df = pd.DataFrame([values])
                pred = model.predict(input_df)[0]
                prob = (model.predict_proba(input_df)[0]
                        if hasattr(model, "predict_proba") else None)

                if pred == 1:
                    st.error(f"⚠️ **FRAUDULENT** transaction detected!"
                             + (f" (probability: {prob[1]:.4f})" if prob is not None else ""))
                else:
                    st.success(f"✅ **LEGITIMATE** transaction."
                               + (f" (fraud probability: {prob[1]:.4f})" if prob is not None else ""))

                # ── Plain-English Explanation ──
                st.subheader("📝 Why this decision?")

                # Try SHAP first for accurate explanations
                explanation_generated = False
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(input_df)
                    # For binary classification, take class-1 values
                    if isinstance(shap_values, list):
                        sv = shap_values[1][0]
                    else:
                        sv = shap_values[0]

                    contrib = pd.Series(sv, index=feature_cols)
                    top_factors = contrib.abs().sort_values(ascending=False).head(5)

                    reasons = []
                    for feat_name in top_factors.index:
                        val = values[feat_name]
                        impact = contrib[feat_name]
                        direction = "increases" if impact > 0 else "decreases"

                        # Human-readable feature descriptions
                        readable = _describe_feature(feat_name, val, impact)
                        reasons.append(readable)

                    if pred == 1:
                        st.markdown("**🔴 This transaction was flagged as fraudulent because:**")
                    else:
                        st.markdown("**🟢 This transaction appears legitimate because:**")

                    for i, reason in enumerate(reasons, 1):
                        st.markdown(f"{i}. {reason}")

                    explanation_generated = True
                except Exception:
                    pass

                # Fallback: simple deviation-based explanation
                if not explanation_generated:
                    sample_data = load_dataset_sample(ds_name, n=5000)
                    if not sample_data.empty:
                        reasons = []
                        for feat_name in feature_cols[:30]:
                            val = values[feat_name]
                            col_data = sample_data[feat_name]
                            mean, std = col_data.mean(), col_data.std()
                            if std > 0:
                                z = (val - mean) / std
                                if abs(z) > 2.0:
                                    readable = _describe_feature(feat_name, val, z)
                                    reasons.append((abs(z), readable))

                        reasons.sort(reverse=True)
                        top_reasons = [r[1] for r in reasons[:5]]

                        if pred == 1:
                            st.markdown("**🔴 This transaction was flagged as fraudulent because:**")
                        else:
                            st.markdown("**🟢 This transaction appears legitimate because:**")

                        if top_reasons:
                            for i, reason in enumerate(top_reasons, 1):
                                st.markdown(f"{i}. {reason}")
                        else:
                            st.info("All feature values are within normal ranges. "
                                    "The model's decision is based on subtle "
                                    "interactions between features.")


# ══════════════════════════════════════════════
# PAGE 4 – Explainability
# ══════════════════════════════════════════════

elif page == "🔍 Explainability":
    st.markdown('<p class="main-header">🔍  Model Explainability</p>', unsafe_allow_html=True)

    for ds in [DS_EUROPEAN, DS_SPARKOV]:
        ds_fig_dir = FIGURES_DIR / ds
        if ds_fig_dir.exists():
            st.subheader(f"{ds.title()} Dataset")
            for img_path in sorted(ds_fig_dir.glob("*.png")):
                st.image(str(img_path), caption=img_path.stem, use_container_width=True)

    # Global figures
    for img_path in sorted(FIGURES_DIR.glob("*.png")):
        if img_path.parent == FIGURES_DIR:
            st.image(str(img_path), caption=img_path.stem, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 5 – 2FA / MFA Simulation
# ══════════════════════════════════════════════

elif page == "🔐 2FA Simulation":
    st.markdown('<p class="main-header">🔐  2FA / MFA Simulation</p>', unsafe_allow_html=True)

    st.markdown("""
    This simulation demonstrates how multi-factor authentication (MFA)
    can serve as an additional fraud-prevention layer, triggered by
    model risk scores.
    """)

    risk_threshold_2fa = st.slider("2FA trigger threshold", 0.0, 1.0, 0.5, 0.05)
    risk_threshold_mfa = st.slider("MFA trigger threshold", 0.0, 1.0, 0.8, 0.05)

    n_transactions = st.number_input("Simulated transactions", 100, 10000, 1000, 100)

    if st.button("🚀 Run Simulation"):
        np.random.seed(RANDOM_SEED)
        risk_scores = np.random.beta(2, 5, n_transactions)  # skewed-low
        # Inject some high-risk
        n_fraud = int(n_transactions * 0.05)
        risk_scores[:n_fraud] = np.random.beta(5, 2, n_fraud)
        np.random.shuffle(risk_scores)

        layer1 = risk_scores < risk_threshold_2fa
        layer2 = (risk_scores >= risk_threshold_2fa) & (risk_scores < risk_threshold_mfa)
        layer3 = risk_scores >= risk_threshold_mfa

        sim_df = pd.DataFrame({
            "risk_score": risk_scores,
            "auth_layer": np.where(layer1, "Standard",
                                   np.where(layer2, "2FA (SMS/Email)", "MFA (Biometric)")),
        })

        col1, col2, col3 = st.columns(3)
        col1.metric("Standard Auth", f"{layer1.sum()} ({layer1.mean()*100:.1f}%)")
        col2.metric("2FA Required", f"{layer2.sum()} ({layer2.mean()*100:.1f}%)")
        col3.metric("MFA Required", f"{layer3.sum()} ({layer3.mean()*100:.1f}%)")

        fig = px.histogram(sim_df, x="risk_score", color="auth_layer",
                           nbins=50, barmode="overlay", opacity=0.7,
                           title="Transaction Risk Scores & Authentication Layers",
                           color_discrete_sequence=["#55A868", "#F0AD4E", "#C44E52"])
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.pie(sim_df, names="auth_layer",
                      title="Authentication Layer Distribution",
                      color_discrete_sequence=["#55A868", "#F0AD4E", "#C44E52"])
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════
# PAGE 6 – Time Series
# ══════════════════════════════════════════════

elif page == "📅 Time Series":
    st.markdown('<p class="main-header">📅  Fraud Trends Over Time</p>', unsafe_allow_html=True)

    ds = st.selectbox("Dataset", [DS_EUROPEAN, DS_SPARKOV], key="ts_ds")
    df = load_dataset_sample(ds, n=50000)

    if df.empty:
        st.warning("Data not found.")
    else:
        if "trans_hour" in df.columns:
            hourly = df.groupby("trans_hour")["target"].agg(["sum", "count"])
            hourly["fraud_rate"] = hourly["sum"] / hourly["count"]
            fig = px.bar(hourly.reset_index(), x="trans_hour", y="fraud_rate",
                         title="Fraud Rate by Hour of Day",
                         labels={"trans_hour": "Hour", "fraud_rate": "Fraud Rate"})
            st.plotly_chart(fig, use_container_width=True)
        elif "Time" in df.columns:
            df["time_bin"] = pd.cut(df["Time"], bins=24)
            binned = df.groupby("time_bin")["target"].agg(["sum", "count"])
            binned["fraud_rate"] = binned["sum"] / binned["count"]
            binned = binned.reset_index()
            binned["time_bin"] = binned["time_bin"].astype(str)
            fig = px.bar(binned, x="time_bin", y="fraud_rate",
                         title="Fraud Rate Over Transaction Time Bins")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No time-based feature found in the processed data.")

        # Amount distribution
        if "amt" in df.columns or "Amount" in df.columns:
            amt_col = "amt" if "amt" in df.columns else "Amount"
            fig2 = px.box(df, x="target", y=amt_col, color="target",
                          title="Transaction Amount by Class",
                          labels={"target": "Class", amt_col: "Amount"},
                          color_discrete_map={0: "#55A868", 1: "#C44E52"})
            st.plotly_chart(fig2, use_container_width=True)
