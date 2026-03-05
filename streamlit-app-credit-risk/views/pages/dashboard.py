import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import pickle
import os

from views.main.layout import page_header
from db.database import init_db, fetch_stats, fetch_recent

# ── Model path ────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_BASE, "models", "credit_risk_model.pkl")

_FEATURE_COLS = [
    "rev_util", "age", "late_30_59", "debt_ratio", "monthly_inc",
    "open_credit", "late_90", "real_estate", "late_60_89", "dependents",
]

RISK_BADGE = {"High": "High", "Medium": "Medium", "Low": "Low"}


@st.cache_resource
def _load_model():
    if not os.path.exists(_MODEL_PATH):
        return None
    with open(_MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Chart builders ─────────────────────────────────────────────────────────────


def _kpi_card(
    container,
    label: str,
    value: str,
    delta: str = None,
    delta_color: str = "normal",
    help: str = None,
):
    """Render a single bordered KPI card.

    Customise any card by changing these parameters:
      label       – title shown above the number
      value       – main numeric/text value
      delta       – change indicator text (e.g. '+2.3% from last month')
      delta_color – 'normal' (green=up), 'inverse' (red=up), or 'off' (grey)
      help        – optional tooltip text on hover
    """
    with container:
        with st.container(border=True):
            st.metric(
                label=label,
                value=value,
                delta=delta,
                delta_color=delta_color,
                help=help,
            )


def _risk_pie(stats: dict) -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=["Low Risk", "Medium Risk", "High Risk"],
            values=[stats["low"], stats["medium"], stats["high"]],
            marker_colors=["#10B981", "#F59E0B", "#EF4444"],
            hole=0.35,
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,}<extra></extra>",
        )
    )
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.15),
        height=300,
    )
    return fig



# ── Main render function ───────────────────────────────────────────────────────


def render():
    init_db()
    page_header(
        "Dashboard",
        "Monitor borrower risk and model performance in real-time",
    )

    # ── Fetch real data ────────────────────────────────────────────────────────
    stats = fetch_stats()
    model = _load_model()
    total = stats["total"] or 1

    # Compute avg default probability from model if available
    avg_default_pct_val = None
    if model is not None:
        recent_all = fetch_recent(limit=stats["total"])
        if not recent_all.empty:
            available = [c for c in _FEATURE_COLS if c in recent_all.columns]
            if len(available) == len(_FEATURE_COLS):
                X = recent_all[_FEATURE_COLS].fillna(0)
                probas = model.predict_proba(X)
                avg_default_pct_val = probas[:, 1].mean() * 100

    avg_default_str = f"{avg_default_pct_val:.1f}%" if avg_default_pct_val is not None else "N/A"

    # Percentage breakdowns for deltas
    high_pct = stats["high"] / total * 100
    med_pct  = stats["medium"] / total * 100
    low_pct  = stats["low"] / total * 100

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    _kpi_card(kpi1, "Total Borrowers",        f"{stats['total']:,}",
              delta=f"{stats['total']:,} records in database",
              help="Total borrowers in database")
    _kpi_card(kpi2, "High Risk Borrowers",     f"{stats['high']:,}",
              delta=f"{high_pct:.1f}% of total",
              delta_color="inverse",
              help="Borrowers classified as High risk")
    _kpi_card(kpi3, "Avg Default Probability", avg_default_str,
              delta=f"Low risk: {low_pct:.1f}%",
              help="Mean predicted default probability across all borrowers")
    _kpi_card(kpi4, "Model Accuracy",          "87.3%",
              delta=f"AUC: 0.86",
              help="Model accuracy from training evaluation")

    # ── Charts ─────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.text("Risk Distribution")
            st.plotly_chart(_risk_pie(stats), config={"displayModeBar": False})

    with chart_col2:
        with st.container(border=True):
            st.text("Risk Breakdown")
            total = stats["total"] or 1
            for level, count, color in [
                ("Low",    stats["low"],    "#10B981"),
                ("Medium", stats["medium"], "#F59E0B"),
                ("High",   stats["high"],   "#EF4444"),
            ]:
                pct = count / total * 100
                st.markdown(f"**{level} Risk** — {count:,} ({pct:.1f}%)")
                st.progress(pct / 100)

    # ── Recent Borrowers ──────────────────────────────────────────────────────
    with st.container(border=True):
        st.text("Recent Borrowers")
        recent_df = fetch_recent(limit=10)
        if recent_df.empty:
            st.info("No borrower records yet.")
        else:
            if model is not None:
                available = [c for c in _FEATURE_COLS if c in recent_df.columns]
                if len(available) == len(_FEATURE_COLS):
                    X = recent_df[_FEATURE_COLS].fillna(0)
                    probs = model.predict_proba(X)
                    recent_df["default_prob"] = np.round(probs[:, 1] * 100, 1)

            display_df = recent_df.copy()
            display_df["risk_level"] = display_df["risk_level"].map(RISK_BADGE)

            col_config = {
                "id":           st.column_config.NumberColumn("ID",          width="small", format="%d"),
                "age":          st.column_config.NumberColumn("Age",         width="small", format="%d"),
                "rev_util":     st.column_config.NumberColumn("Rev. Util",   width="small", format="%.3f"),
                "debt_ratio":   st.column_config.NumberColumn("Debt Ratio",  width="small", format="%.3f"),
                "monthly_inc":  st.column_config.NumberColumn("Monthly Inc", width="small", format="%.0f"),
                "risk_level":   st.column_config.TextColumn("Risk Level",    width="small"),
            }
            if "default_prob" in display_df.columns:
                col_config["default_prob"] = st.column_config.ProgressColumn(
                    "Default Prob %", min_value=0, max_value=100, format="%.1f%%"
                )

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                column_config=col_config,
            )
