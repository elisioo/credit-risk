import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from views.main.layout import page_header
from db.dataset_generator import generate_dataset
from db.uploaded_datasets import insert_dataframe, fetch_all as fetch_uploaded, row_count as uploaded_count, clear_all as clear_uploaded
from db.database import bulk_insert as bulk_insert_borrowers
from db.predictions_history import log_single, log_batch, fetch_history, total_count, clear_history

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

    tab_single, tab_batch, tab_generate, tab_uploaded, tab_history = st.tabs(
        ["Single Prediction", "Batch Prediction (CSV)", "Generate Test Dataset", "Uploaded Datasets", "History"]
    )


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
| **DLQ 2 Years (`dlq_2yrs`)** |  **Target variable.** This is what the model predicts. `1` means the borrower is likely to **default** (miss payments seriously) within 2 years. `0` means they are likely to **pay on time**. |
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
            with st.spinner("Predicting credit risk..."):
                row = pd.DataFrame(
                    [[rev_util, int(age), int(late_30_59), debt_ratio, monthly_inc,
                      int(open_credit), int(late_90), int(real_estate), int(late_60_89), int(dependents)]],
                    columns=FEATURE_COLS,
                )
                pred = model.predict(row)[0]
                proba = model.predict_proba(row)[0]
                log_single(
                    rev_util=rev_util, age=int(age), late_30_59=int(late_30_59),
                    debt_ratio=debt_ratio, monthly_inc=monthly_inc,
                    open_credit=int(open_credit), late_90=int(late_90),
                    real_estate=int(real_estate), late_60_89=int(late_60_89),
                    dependents=int(dependents),
                    dlq_2yrs=int(pred), default_prob=float(proba[1] * 100),
                )
                st.session_state["single_result"] = {
                    "row": row,
                    "pred": int(pred),
                    "proba": proba,
                }

        if "single_result" in st.session_state:
            res = st.session_state["single_result"]
            _render_single_result(res["pred"], res["proba"])

            # Show input row with predicted dlq_2yrs highlighted
            display_row = res["row"].copy()
            display_row["dlq_2yrs"] = res["pred"]
            styled = display_row.style.applymap(
                lambda _: "background-color: #fff3cd; font-weight: bold;",
                subset=["dlq_2yrs"],
            )
            st.subheader("Borrower Data with Prediction")
            st.dataframe(styled, use_container_width=True, hide_index=True)

            if st.button("Save", key="save_single"):
                save_df = display_row.copy()
                bulk_insert_borrowers(save_df)
                st.success("Borrower saved to Borrower Data!")


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
                with st.spinner("Running predictions, please wait..."):
                    preds = model.predict(df_input)
                    probas = model.predict_proba(df_input)

                    results = df.copy()
                    results["dlq_2yrs"] = preds
                    results["prediction"] = np.where(preds == 1, "DEFAULT", "NO DEFAULT")
                    results["default_prob_%"] = np.round(probas[:, 1] * 100, 1)
                    results["risk_level"] = results["default_prob_%"].apply(
                        lambda p: "Low" if p < 30 else ("Moderate" if p < 60 else "High")
                    )
                    st.session_state["batch_results"] = results
                    log_batch(results)

            if "batch_results" in st.session_state:
                results = st.session_state["batch_results"]

                with st.spinner("Loading results..."):
                    # Summary metrics
                    n = len(results)
                    n_default = (results["dlq_2yrs"] == 1).sum()
                    avg_prob = results["default_prob_%"].mean()

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Borrowers", f"{n:,}")
                    m2.metric("Predicted Defaults", f"{n_default:,}")
                    m3.metric("Avg Default Prob", f"{avg_prob:.1f}%")

                    # Highlight predicted columns
                    highlight_cols = ["dlq_2yrs", "prediction", "default_prob_%", "risk_level"]
                    styled_results = results.style.applymap(
                        lambda _: "background-color: #fff3cd; font-weight: bold;",
                        subset=[c for c in highlight_cols if c in results.columns],
                    )
                    st.dataframe(
                        styled_results,
                        hide_index=True,
                        use_container_width=True,
                    )

                # Download results
                csv_out = results.to_csv(index=False)
                dl_col, save_col = st.columns(2)
                with dl_col:
                    st.download_button(
                        "⬇ Download Results CSV",
                        data=csv_out,
                        file_name="prediction_results.csv",
                        mime="text/csv",
                    )
                with save_col:
                    if st.button("Save to Borrower Data", type="primary", key="btn_save_borrower"):
                        count = bulk_insert_borrowers(results)
                        st.success(f"Saved **{count:,}** rows to Borrower Data (with predictions)!")


    #  TAB 3 – Generate synthetic test dataset

    with tab_generate:
        st.subheader("Generate Synthetic Test Dataset")
        st.markdown(
            "Generate a synthetic dataset with only the **feature columns** "
            "(`rev_util`, `age`, `late_30_59`, `debt_ratio`, `monthly_inc`, "
            "`open_credit`, `late_90`, `real_estate`, `late_60_89`, `dependents`). "
            "The `dlq_2yrs` column is **not included** — download the CSV, "
            "then upload it in the **Batch Prediction** tab to let the model predict it. "
            "Once predicted, you can save the results to **Uploaded Datasets**."
        )

        col_rows, col_seed = st.columns(2)
        n_rows = col_rows.number_input(
            "Number of rows to generate",
            min_value=1, max_value=100_000, value=100, step=10,
        )
        use_seed = col_seed.checkbox("Set random seed (for reproducibility)", value=False)
        seed_val = col_seed.number_input("Seed", min_value=0, value=42, disabled=not use_seed)

        if st.button("Generate Dataset", type="primary", key="btn_generate"):
            with st.spinner(f"Generating {int(n_rows):,} rows..."):
                seed = int(seed_val) if use_seed else None
                gen_df = generate_dataset(int(n_rows), seed=seed)
                st.session_state["generated_df"] = gen_df

        if "generated_df" in st.session_state:
            gen_df = st.session_state["generated_df"]
            st.success(f"Generated **{len(gen_df):,}** rows")
            with st.spinner("Loading table..."):
                st.dataframe(gen_df, use_container_width=True, hide_index=True)

            csv_data = gen_df.to_csv(index=False)
            st.download_button(
                "⬇ Download Generated CSV",
                data=csv_data,
                file_name="generated_credit_risk_dataset.csv",
                mime="text/csv",
                key="dl_generated",
            )
            st.info("Upload this CSV in the **Batch Prediction** tab to run predictions, then save the results to Uploaded Datasets.")


    #  TAB 4 – Uploaded / stored datasets

    with tab_uploaded:
        st.subheader("Upload & Store Datasets")
        st.markdown(
            "Upload a CSV file to store it in the application database. "
            "This keeps your custom datasets separate from the original benchmark data."
        )

        upload_csv = st.file_uploader(
            "Choose a CSV file to upload",
            type=["csv"],
            key="upload_store",
        )

        if upload_csv is not None:
            try:
                up_df = pd.read_csv(upload_csv)
                up_df.columns = [c.strip().lower() for c in up_df.columns]
            except Exception as e:
                st.error(f"Could not read the CSV file: {e}")
                up_df = None

            if up_df is not None:
                st.caption(f"Preview – **{len(up_df):,}** rows")
                st.dataframe(up_df.head(50), use_container_width=True, hide_index=True)

                if st.button("Upload to Database", type="primary", key="btn_store_csv"):
                    count = insert_dataframe(up_df)
                    st.success(f"Stored **{count:,}** rows successfully!")

        st.divider()
        st.subheader("Stored Dataset Records")

        total = uploaded_count()
        st.caption(f"Total stored rows: **{total:,}**")

        if total > 0:
            stored_df = fetch_uploaded()
            st.dataframe(stored_df, use_container_width=True, hide_index=True)

            sc1, sc2 = st.columns(2)
            with sc1:
                csv_stored = stored_df.to_csv(index=False)
                st.download_button(
                    "⬇ Download Stored Dataset",
                    data=csv_stored,
                    file_name="stored_credit_risk_dataset.csv",
                    mime="text/csv",
                    key="dl_stored",
                )
            with sc2:
                if st.button("Clear All Stored Data", type="secondary", key="btn_clear_stored"):
                    clear_uploaded()
                    st.success("All stored data cleared!")
                    st.rerun()
        else:
            st.info("No datasets uploaded yet. Generate or upload a CSV above.")


    #  TAB 5 – Prediction History

    with tab_history:
        st.subheader("Prediction History")
        st.markdown(
            "A persistent log of every prediction made in this session and previous sessions, "
            "with date and time."
        )

        total_hist = total_count()

        h_info, h_clear = st.columns([5, 1])
        h_info.caption(f"Total predictions logged: **{total_hist:,}**")
        if h_clear.button("🗑 Clear History", type="secondary", key="btn_clear_hist"):
            clear_history()
            st.success("History cleared.")
            st.rerun()

        if total_hist == 0:
            st.info("No predictions yet. Run a Single or Batch prediction to start logging.")
        else:
            # Filter controls
            fc1, fc2, fc3 = st.columns(3)
            src_filter = fc1.selectbox("Source", ["All", "single", "batch"], key="hist_src")
            pred_filter = fc2.selectbox("Prediction", ["All", "DEFAULT", "NO DEFAULT"], key="hist_pred")
            risk_filter = fc3.selectbox("Risk Level", ["All", "High", "Moderate", "Low"], key="hist_risk")

            hist_df = fetch_history(limit=1000)

            if src_filter != "All":
                hist_df = hist_df[hist_df["source"] == src_filter]
            if pred_filter != "All":
                hist_df = hist_df[hist_df["prediction"] == pred_filter]
            if risk_filter != "All":
                hist_df = hist_df[hist_df["risk_level"] == risk_filter]

            st.caption(f"Showing **{len(hist_df):,}** records")

            def _color_prediction(val):
                if val == "DEFAULT":
                    return "background-color: #fde8e8; color: #c0392b; font-weight: bold;"
                elif val == "NO DEFAULT":
                    return "background-color: #e8f8f0; color: #1e8449; font-weight: bold;"
                return ""

            def _color_risk(val):
                mapping = {
                    "High":     "background-color: #fde8e8; color: #c0392b;",
                    "Moderate": "background-color: #fff3cd; color: #856404;",
                    "Low":      "background-color: #e8f8f0; color: #1e8449;",
                }
                return mapping.get(val, "")

            styled_hist = hist_df.style \
                .applymap(_color_prediction, subset=["prediction"]) \
                .applymap(_color_risk, subset=["risk_level"])

            st.dataframe(
                styled_hist,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "id":            st.column_config.NumberColumn("ID",          width="small",  format="%d"),
                    "predicted_at":  st.column_config.TextColumn("Date & Time",   width="medium"),
                    "source":        st.column_config.TextColumn("Source",        width="small"),
                    "prediction":    st.column_config.TextColumn("Prediction",    width="medium"),
                    "risk_level":    st.column_config.TextColumn("Risk Level",    width="small"),
                    "default_prob":  st.column_config.NumberColumn("Default Prob %", width="small", format="%.1f"),
                    "dlq_2yrs":      st.column_config.NumberColumn("dlq_2yrs",    width="small",  format="%d"),
                    "rev_util":      st.column_config.NumberColumn("Rev. Util",   width="small",  format="%.3f"),
                    "age":           st.column_config.NumberColumn("Age",         width="small",  format="%d"),
                    "debt_ratio":    st.column_config.NumberColumn("Debt Ratio",  width="small",  format="%.3f"),
                    "monthly_inc":   st.column_config.NumberColumn("Monthly Inc", width="medium", format="%.0f"),
                },
            )

            csv_hist = hist_df.to_csv(index=False)
            st.download_button(
                "⬇ Download History CSV",
                data=csv_hist,
                file_name="prediction_history.csv",
                mime="text/csv",
                key="dl_history",
            )
