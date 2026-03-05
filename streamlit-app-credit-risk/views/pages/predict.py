import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from views.main.layout import page_header

# Model path 
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(_BASE, "models", "credit_risk_model.pkl")

FEATURE_COLS = [
    "rev_util", "age", "late_30_59", "debt_ratio", "monthly_inc",
    "open_credit", "late_90", "real_estate", "late_60_89", "dependents",
]


@st.cache_resource
def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _classify(default_pct: float) -> tuple[str, str, str]:
    """Return (emoji_label, color, advice) based on default probability."""
    if default_pct < 30:
        return "LOW RISK", "green", "Likely safe to approve the loan."
    elif default_pct < 60:
        return "MODERATE RISK", "orange", "Consider higher interest rate or reduced limit."
    else:
        return "HIGH RISK", "red", "Recommend rejecting or requiring collateral."


def _render_single_result(pred, proba):
    """Display prediction result for a single borrower."""
    default_pct = proba[1] * 100
    no_default_pct = proba[0] * 100
    label, color, advice = _classify(default_pct)

    st.divider()
    st.subheader("Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", "DEFAULT" if pred == 1 else "NO DEFAULT")
    c2.metric("Default Probability", f"{default_pct:.1f}%")
    c3.metric("No-Default Probability", f"{no_default_pct:.1f}%")

    st.markdown(f"**Risk Level:** :{color}[{label}]")
    st.progress(default_pct / 100)
    st.info(f"**Advice:** {advice}")


def render():
    model = _load_model()
    page_header("Predictions", "Run real-time credit risk predictions")

    if model is None:
        st.error(
            f"Model file not found at `{MODEL_PATH}`.\n\n"
            "Please run the training notebook first to generate `credit_risk_model.pkl`."
        )
        return

    tab_single, tab_batch = st.tabs(["Single Prediction", "Batch Prediction (CSV)"])


    #  TAB 1 – Single borrower form

    with tab_single:
        with st.expander("What do these fields mean?", expanded=False):
            st.markdown(
                """
| Field | Description |
|---|---|
| **Revolving Utilization** | Ratio of total credit card balance to total credit limit (0 = no usage, 1 = maxed out) |
| **Age** | Borrower's age in years |
| **Late 30-59 Days** | Number of times 30-59 days past due in the last 2 years |
| **Debt Ratio** | Monthly debt payments divided by monthly gross income |
| **Monthly Income** | Borrower's gross monthly income in dollars |
| **Open Credit Lines** | Number of currently open loans or credit lines |
| **Late 90+ Days** | Number of times 90+ days past due in the last 2 years |
| **Real Estate Loans** | Number of mortgage or real estate loans |
| **Late 60-89 Days** | Number of times 60-89 days past due in the last 2 years |
| **Dependents** | Number of dependents in the household |
"""
            )

        with st.form("predict_form"):
            st.subheader("Enter Borrower Details")
            c1, c2, c3 = st.columns(3)

            rev_util    = c1.number_input("Revolving Utilization (0–1)", min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.3f")
            age         = c2.number_input("Age", min_value=18, max_value=100, value=35)
            late_30_59  = c3.number_input("Late 30-59 Days", min_value=0, value=0)
            debt_ratio  = c1.number_input("Debt Ratio", min_value=0.0, value=0.3, step=0.01, format="%.3f")
            monthly_inc = c2.number_input("Monthly Income ($)", min_value=0.0, value=5000.0, step=100.0)
            open_credit = c3.number_input("Open Credit Lines", min_value=0, value=5)
            late_90     = c1.number_input("Late 90+ Days", min_value=0, value=0)
            real_estate = c2.number_input("Real Estate Loans", min_value=0, value=1)
            late_60_89  = c3.number_input("Late 60-89 Days", min_value=0, value=0)
            dependents  = c1.number_input("Dependents", min_value=0, value=0)

            submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

        if submitted:
            row = pd.DataFrame(
                [[rev_util, int(age), int(late_30_59), debt_ratio, monthly_inc,
                  int(open_credit), int(late_90), int(real_estate), int(late_60_89), int(dependents)]],
                columns=FEATURE_COLS,
            )
            pred = model.predict(row)[0]
            proba = model.predict_proba(row)[0]
            _render_single_result(pred, proba)


    #  TAB 2 – Batch CSV upload

    with tab_batch:
        st.subheader("Upload CSV for Batch Prediction")
        st.markdown(
            "The CSV must contain these columns (in any order):\n"
            f"```\n{', '.join(FEATURE_COLS)}\n```"
        )

        # Downloadable template
        template_df = pd.DataFrame(columns=FEATURE_COLS)
        st.download_button(
            "⬇ Download CSV Template",
            data=template_df.to_csv(index=False),
            file_name="predict_template.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded is not None:
            try:
                df = pd.read_csv(uploaded)
                df.columns = [c.strip().lower() for c in df.columns]
            except Exception as e:
                st.error(f"Could not read the CSV file: {e}")
                return

            missing = [c for c in FEATURE_COLS if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: **{', '.join(missing)}**")
                return

            df_input = df[FEATURE_COLS].fillna(0)
            st.caption(f"Loaded **{len(df_input):,}** rows")

            if st.button("Run Batch Prediction", type="primary"):
                preds = model.predict(df_input)
                probas = model.predict_proba(df_input)

                results = df.copy()
                results["prediction"] = np.where(preds == 1, "DEFAULT", "NO DEFAULT")
                results["default_prob_%"] = np.round(probas[:, 1] * 100, 1)
                results["risk_level"] = results["default_prob_%"].apply(
                    lambda p: "🟢 Low" if p < 30 else ("🟡 Moderate" if p < 60 else "🔴 High")
                )

                # Summary metrics
                n = len(results)
                n_default = (preds == 1).sum()
                avg_prob = probas[:, 1].mean() * 100

                m1, m2, m3 = st.columns(3)
                m1.metric("Total Borrowers", f"{n:,}")
                m2.metric("Predicted Defaults", f"{n_default:,}")
                m3.metric("Avg Default Prob", f"{avg_prob:.1f}%")

                st.dataframe(
                    results,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "default_prob_%": st.column_config.ProgressColumn(
                            "Default Prob %", min_value=0, max_value=100, format="%.1f%%"
                        ),
                    },
                )

                # Download results
                csv_out = results.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv_out,
                    file_name="prediction_results.csv",
                    mime="text/csv",
                )
