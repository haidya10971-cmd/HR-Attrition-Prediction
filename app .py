"""
app.py — HR Attrition Decision Support System
Production-ready Streamlit application.

Fixes applied vs original:
  1. Dynamic threshold  — loaded from artifact, not hardcoded at 0.50
  2. High-risk logic    — uses optimal_threshold, not 50%
  3. Progress bar crash — st.progress(risk / 100.0) instead of st.progress(int(risk))
  4. Feature alignment  — builds a fully ordered DataFrame before scaler.transform()
                          to eliminate sklearn feature-name mismatch warnings
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HR Attrition DSS",
    page_icon="🏢",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Load model artifact
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the pickled artifact bundle produced by §1.13 of the notebook."""
    with open("hr_model.pkl", "rb") as f:
        return pickle.load(f)


try:
    art = load_model()
except FileNotFoundError:
    st.error(
        "**hr_model.pkl not found.**\n\n"
        "Run the training notebook (§1.13) first to generate the artifact, "
        "then restart this app."
    )
    st.stop()

# ── Unpack all required keys ─────────────────────────────────────────────────
model             = art["model"]
scaler            = art["scaler"]
encoders          = art["encoders"]        # {column_name: fitted LabelEncoder}
features          = art["features"]        # ordered list — matches scaler
fi                = art["fi"]              # sorted feature-importance DataFrame
medians           = art["medians"]         # {col: median} from training set
modes             = art["modes"]           # {col: mode}   from training set
scores            = art["scores"]          # {model_name: {acc, prec, rec, f1, auc}}
name              = art["model_name"]      # e.g. "XGBoost Classifier"

# FIX 1 — dynamic threshold (fall back to 0.5 if key missing)
optimal_threshold = float(art.get("threshold", 0.5))


# ---------------------------------------------------------------------------
# Feature lists
# ---------------------------------------------------------------------------
# Engineered features that are not exposed in the sidebar form because
# they are computed automatically from other inputs.
_ENGINEERED = {"SatisfactionScore", "IncomePerExp", "LoyaltyRatio", "PromotionLag",
               # legacy names kept for backward compatibility
               "AvgSatisfaction", "IncomePerYear"}

form_features = [f for f in features if f not in _ENGINEERED]
cat_features  = [f for f in form_features if f in encoders]
num_features  = [f for f in form_features if f not in encoders]


# ---------------------------------------------------------------------------
# Feature engineering  (mirrors §1.6 of the notebook exactly)
# ---------------------------------------------------------------------------
def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns.  Input df must already have raw feature values."""
    df = df.copy()

    # SatisfactionScore (notebook name)
    sat_cols = [
        "JobSatisfaction", "EnvironmentSatisfaction",
        "RelationshipSatisfaction", "WorkLifeBalance",
    ]
    present = [c for c in sat_cols if c in df.columns]
    if present:
        df["SatisfactionScore"] = df[present].mean(axis=1)

    # AvgSatisfaction (app.py legacy name — kept for safety)
    if "SatisfactionScore" in df.columns:
        df["AvgSatisfaction"] = df["SatisfactionScore"]

    # IncomePerExp / IncomePerYear
    if "MonthlyIncome" in df.columns and "TotalWorkingYears" in df.columns:
        val = df["MonthlyIncome"] / (df["TotalWorkingYears"] + 1)
        df["IncomePerExp"]  = val
        df["IncomePerYear"] = val   # legacy alias

    # LoyaltyRatio
    if "YearsAtCompany" in df.columns and "TotalWorkingYears" in df.columns:
        df["LoyaltyRatio"] = df["YearsAtCompany"] / (df["TotalWorkingYears"] + 1)

    # PromotionLag
    if "YearsSinceLastPromotion" in df.columns and "YearsAtCompany" in df.columns:
        df["PromotionLag"] = (
            df["YearsSinceLastPromotion"] / (df["YearsAtCompany"] + 1)
        )

    return df


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_employee(inputs: dict) -> float:
    """
    Run inference for a single employee.

    Parameters
    ----------
    inputs : dict
        Raw values collected from the sidebar form.

    Returns
    -------
    float
        Attrition probability as a percentage (0–100).
    """
    # ── Step 1: build a row with all form features ────────────────────────
    row: dict = {}
    for feat in form_features:
        if feat in encoders:
            le  = encoders[feat]
            val = str(inputs.get(feat, modes.get(feat, le.classes_[0])))
            val = val if val in le.classes_ else le.classes_[0]
            row[feat] = int(le.transform([val])[0])
        else:
            row[feat] = float(inputs.get(feat, medians.get(feat, 0.0)))

    # ── Step 2: create DataFrame, engineer derived features ───────────────
    df = pd.DataFrame([row])
    df = _engineer(df)

    # ── Step 3: ensure all training features are present (fill missing) ───
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    # ── FIX 4: reorder to exact training layout before scaler.transform() ─
    # Passing an ordered DataFrame (not a numpy array) prevents the
    # sklearn "Feature names must match" UserWarning / ValueError.
    X_ordered = df[features].fillna(0.0)

    # ── Step 4: scale and predict ─────────────────────────────────────────
    X_scaled = scaler.transform(X_ordered)
    prob      = float(model.predict_proba(X_scaled)[0, 1])
    return round(prob * 100, 1)


# ---------------------------------------------------------------------------
# Sidebar — employee input form
# ---------------------------------------------------------------------------
st.sidebar.title("👤 Employee Details")

inputs: dict = {}

st.sidebar.subheader("Personal & Role")
for feat in cat_features:
    opts    = list(encoders[feat].classes_)
    default = modes.get(feat, opts[0])
    idx     = opts.index(default) if default in opts else 0
    inputs[feat] = st.sidebar.selectbox(feat, opts, index=idx)

st.sidebar.subheader("Work Details")
for feat in num_features:
    default       = float(medians.get(feat, 0.0))
    inputs[feat]  = st.sidebar.number_input(feat, value=default, step=1.0)

predict_btn = st.sidebar.button(
    "🔍 Predict Attrition Risk",
    type="primary",
    use_container_width=True,
)

# Show threshold info in sidebar
st.sidebar.divider()
st.sidebar.caption(
    f"**Optimal threshold:** {optimal_threshold:.2f} "
    f"(F1-maximised on test set)"
)


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
st.title("🏢 HR Attrition — Decision Support System")
st.caption(
    f"Model: **{name}** | "
    f"Accuracy: {scores[name]['acc']*100:.1f}% | "
    f"AUC: {scores[name]['auc']*100:.1f}% | "
    f"Threshold: {optimal_threshold:.2f}"
)
st.divider()

# ── Default view (no prediction yet) ────────────────────────────────────────
if not predict_btn:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model",     name)
    col2.metric("Accuracy",  f"{scores[name]['acc']*100:.1f}%")
    col3.metric("AUC Score", f"{scores[name]['auc']*100:.1f}%")
    col4.metric("Threshold", f"{optimal_threshold:.2f}")

    st.divider()
    st.subheader("📊 Top Attrition Risk Factors")
    chart = fi.head(10).set_index("Feature")["Importance"].sort_values()
    st.bar_chart(chart, color=["#e74c3c"])

    st.info(
        "Fill in the employee details in the **left sidebar** and click "
        "**Predict Attrition Risk**."
    )

# ── Prediction view ──────────────────────────────────────────────────────────
else:
    try:
        risk = predict_employee(inputs)

        # FIX 2 — use optimal_threshold, not hardcoded 0.50
        high_risk = risk >= (optimal_threshold * 100)

        # Result banner
        if high_risk:
            st.error(f"### 🔴 HIGH RISK — {risk}% chance of leaving")
        else:
            st.success(f"### 🟢 LOW RISK — {risk}% chance of leaving")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Risk Score", f"{risk}%")
            st.metric(
                "Threshold",
                f"{optimal_threshold*100:.0f}%",
                delta=f"{risk - optimal_threshold*100:+.1f}pp",
                delta_color="inverse",
            )

        # FIX 3 — st.progress() accepts float 0.0–1.0 (not int 0–100)
        with col2:
            st.write("**Risk Level**")
            st.progress(risk / 100.0)

        st.divider()

        # ── Top 5 feature importance ───────────────────────────────────────
        st.subheader("📋 Top 5 Global Risk Factors")
        top5 = fi.head(5).copy().reset_index(drop=True)
        top5.index = top5.index + 1   # 1-based rank
        st.dataframe(
            top5[["Feature", "Importance"]],
            use_container_width=True,
        )

        st.divider()

        # ── HR recommendations ─────────────────────────────────────────────
        st.subheader("💡 HR Recommendations")
        tips: list[str] = []

        ot   = str(inputs.get("OverTime", "")).lower()
        jsat = float(inputs.get("JobSatisfaction",        medians.get("JobSatisfaction", 3)))
        yslp = float(inputs.get("YearsSinceLastPromotion", medians.get("YearsSinceLastPromotion", 0)))
        wlb  = float(inputs.get("WorkLifeBalance",         medians.get("WorkLifeBalance", 3)))
        env  = float(inputs.get("EnvironmentSatisfaction", medians.get("EnvironmentSatisfaction", 3)))
        inc  = float(inputs.get("MonthlyIncome",           medians.get("MonthlyIncome", 0)))
        dist = float(inputs.get("DistanceFromHome",        medians.get("DistanceFromHome", 0)))

        if "yes" in ot:
            tips.append("⏰ **Reduce overtime** — chronic overwork is the single "
                        "strongest attrition predictor in this dataset.")
        if jsat <= 2:
            tips.append("😞 **Address job satisfaction** — schedule a 1-on-1 review "
                        "to understand concerns and explore role adjustments.")
        if yslp > 3:
            tips.append("🚀 **Consider a promotion or stretch assignment** — this "
                        "employee has not been promoted in 3+ years.")
        if wlb <= 2:
            tips.append("🏠 **Improve work-life balance** — offer flexible hours, "
                        "remote work, or additional leave days.")
        if env <= 2:
            tips.append("🌱 **Address work environment** — consider team-building "
                        "initiatives or a change of team/location.")
        if inc < float(medians.get("MonthlyIncome", inc + 1)) * 0.75:
            tips.append("💰 **Review compensation** — monthly income is notably below "
                        "the company median; a market-rate adjustment may help retention.")
        if dist > 20:
            tips.append("🚗 **Long commute detected** — consider remote/hybrid options "
                        "or a transport allowance.")

        if not tips:
            tips.append("✅ No major risk flags detected. Maintain regular check-ins, "
                        "recognition programmes, and growth opportunities.")

        for tip in tips:
            st.markdown(f"- {tip}")

        st.divider()

        # ── All model scores ───────────────────────────────────────────────
        with st.expander("📊 All Model Performance Scores"):
            scores_df = pd.DataFrame(
                [
                    {
                        "Model":     m,
                        "Accuracy":  f"{v['acc']*100:.1f}%",
                        "Precision": f"{v['prec']*100:.1f}%",
                        "Recall":    f"{v['rec']*100:.1f}%",
                        "F1":        f"{v['f1']*100:.1f}%",
                        "AUC":       f"{v['auc']*100:.1f}%",
                    }
                    for m, v in scores.items()
                ]
            )
            st.dataframe(scores_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"**Prediction error:** {exc}")
        st.exception(exc)
