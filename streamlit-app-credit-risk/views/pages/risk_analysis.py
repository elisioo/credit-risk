import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from views.main.layout import page_header
from db.database import init_db, fetch_all

RISK_COLORS = {"Low": "#10B981", "Medium": "#F59E0B", "High": "#EF4444"}
RISK_ORDER  = ["Low", "Medium", "High"]


@st.cache_data(ttl=60)
def _load_data() -> pd.DataFrame:
    return fetch_all()


# ── Chart helpers 

def _risk_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_level"].value_counts().reindex(RISK_ORDER, fill_value=0)
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        marker_colors=[RISK_COLORS[r] for r in RISK_ORDER],
        hole=0.45,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="h", y=-0.15),
        height=300,
    )
    return fig


def _age_distribution(df: pd.DataFrame) -> go.Figure:
    bins   = [17, 30, 45, 60, 120]
    labels = ["18–30", "31–45", "46–60", "61+"]
    df2 = df.copy()
    df2["age_group"] = pd.cut(df2["age"], bins=bins, labels=labels, right=True)
    grouped = (
        df2.groupby(["age_group", "risk_level"], observed=True)
        .size()
        .reset_index(name="count")
    )
    fig = go.Figure()
    for risk in RISK_ORDER:
        sub = grouped[grouped["risk_level"] == risk]
        fig.add_trace(go.Bar(
            name=risk, x=sub["age_group"].astype(str), y=sub["count"],
            marker_color=RISK_COLORS[risk],
            hovertemplate=f"{risk}: %{{y:,}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Age Group", yaxis_title="Borrowers",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def _scatter_debt_rev(df: pd.DataFrame) -> go.Figure:
    # Sample for performance (max 3 000 points per group)
    fig = go.Figure()
    for risk in RISK_ORDER:
        sub = df[df["risk_level"] == risk].sample(
            n=min(3000, len(df[df["risk_level"] == risk])), random_state=42
        )
        fig.add_trace(go.Scatter(
            x=sub["debt_ratio"], y=sub["rev_util"],
            mode="markers", name=risk,
            marker=dict(color=RISK_COLORS[risk], size=4, opacity=0.45),
            hovertemplate="Debt Ratio: %{x:.3f}<br>Rev Util: %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Debt Ratio", yaxis_title="Revolving Utilization",
        legend=dict(orientation="h", y=-0.25),
        xaxis=dict(range=[0, min(df["debt_ratio"].quantile(0.99), 2)]),
        yaxis=dict(range=[0, 1.05]),
    )
    return fig


def _late_payment_bar(df: pd.DataFrame) -> go.Figure:
    cols   = ["late_30_59", "late_60_89", "late_90"]
    labels = {"late_30_59": "Late 30-59", "late_60_89": "Late 60-89", "late_90": "Late 90+"}
    colors = {"late_30_59": "#FCD34D", "late_60_89": "#FB923C", "late_90": "#EF4444"}
    agg = df.groupby("risk_level")[cols].mean().reindex(RISK_ORDER)
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Bar(
            name=labels[col],
            x=agg.index.tolist(),
            y=agg[col].round(3),
            marker_color=colors[col],
            hovertemplate=f"{labels[col]}: %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis_title="Risk Level", yaxis_title="Avg Late Payments",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def _income_box(df: pd.DataFrame) -> go.Figure:
    p99 = df["monthly_inc"].quantile(0.99)
    fig = go.Figure()
    for risk in RISK_ORDER:
        sub = df[df["risk_level"] == risk]["monthly_inc"].clip(upper=p99)
        fig.add_trace(go.Box(
            y=sub, name=risk,
            marker_color=RISK_COLORS[risk],
            boxpoints=False,
            hovertemplate="%{y:.0f}<extra></extra>",
        ))
    fig.update_layout(
        height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="Monthly Income ($)",
        showlegend=False,
    )
    return fig


def _avg_metrics_bar(df: pd.DataFrame) -> go.Figure:
    agg = df.groupby("risk_level")[["debt_ratio", "rev_util"]].mean().reindex(RISK_ORDER)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Avg Debt Ratio", x=agg.index.tolist(), y=agg["debt_ratio"].round(3),
        marker_color="#6366F1",
        hovertemplate="Debt Ratio: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Avg Rev Util", x=agg.index.tolist(), y=agg["rev_util"].round(3),
        marker_color="#EC4899",
        hovertemplate="Rev Util: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        barmode="group", height=300,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis_title="Average Value",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def _segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("risk_level").agg(
        Count          =("id",          "count"),
        Avg_Age        =("age",         "mean"),
        Avg_Income     =("monthly_inc", "mean"),
        Avg_Debt_Ratio =("debt_ratio",  "mean"),
        Avg_Rev_Util   =("rev_util",    "mean"),
        Pct_Delinquent =("dlq_2yrs",    "mean"),
    ).reindex(RISK_ORDER)
    agg["Pct_Delinquent"] = (agg["Pct_Delinquent"] * 100).round(1)
    agg["Avg_Age"]         = agg["Avg_Age"].round(1)
    agg["Avg_Income"]      = agg["Avg_Income"].round(0)
    agg["Avg_Debt_Ratio"]  = agg["Avg_Debt_Ratio"].round(3)
    agg["Avg_Rev_Util"]    = agg["Avg_Rev_Util"].round(3)
    return agg.reset_index().rename(columns={"risk_level": "Risk Level"})


# ── Render ────────────────────────────────────────────────────────────────────

def render():
    init_db()
    page_header("Risk Analysis", "Deep-dive into portfolio risk segments")

    df = _load_data()
    if df.empty:
        st.info("No borrower data available yet. Add borrowers on the Borrower Data page.")
        return

    total  = len(df)
    counts = df["risk_level"].value_counts().to_dict()
    high   = counts.get("High",   0)
    med    = counts.get("Medium", 0)
    low    = counts.get("Low",    0)

    # ── KPI row ───────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, delta, dcolor in [
        (k1, "Total Borrowers", f"{total:,}",  f"{total:,} records",          "off"),
        (k2, "High Risk",       f"{high:,}",   f"{high/total*100:.1f}% of portfolio", "inverse"),
        (k3, "Medium Risk",     f"{med:,}",    f"{med/total*100:.1f}% of portfolio",  "off"),
        (k4, "Low Risk",        f"{low:,}",    f"{low/total*100:.1f}% of portfolio",  "normal"),
    ]:
        with col:
            with st.container(border=True):
                st.metric(label=label, value=val, delta=delta, delta_color=dcolor)

    # ── Row 1: Risk donut | Age breakdown ─────────────────────────────────────
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        with st.container(border=True):
            st.caption("Risk Distribution")
            st.plotly_chart(_risk_donut(df), use_container_width=True, config={"displayModeBar": False})
    with r1c2:
        with st.container(border=True):
            st.caption("Age Group Breakdown by Risk")
            st.plotly_chart(_age_distribution(df), use_container_width=True, config={"displayModeBar": False})

    # ── Row 2: Late payments 
    with st.container(border=True):
        st.caption("Avg Late Payments by Risk Level")
        st.plotly_chart(_late_payment_bar(df), use_container_width=True, config={"displayModeBar": False})

    # ── Row 3: Income box | Avg metrics bar
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        with st.container(border=True):
            st.caption("Monthly Income Distribution by Risk")
            st.plotly_chart(_income_box(df), use_container_width=True, config={"displayModeBar": False})
    with r3c2:
        with st.container(border=True):
            st.caption("Avg Debt Ratio & Rev Utilization by Risk Level")
            st.plotly_chart(_avg_metrics_bar(df), use_container_width=True, config={"displayModeBar": False})

    # ── Segment summary table
    with st.container(border=True):
        st.caption("Risk Segment Summary")
        st.dataframe(
            _segment_summary(df),
            hide_index=True,
            use_container_width=True,
            column_config={
                "Risk Level":     st.column_config.TextColumn("Risk Level"),
                "Count":          st.column_config.NumberColumn("Count",          format="%d"),
                "Avg_Age":        st.column_config.NumberColumn("Avg Age",        format="%.1f"),
                "Avg_Income":     st.column_config.NumberColumn("Avg Income",     format="$%.0f"),
                "Avg_Debt_Ratio": st.column_config.NumberColumn("Avg Debt Ratio", format="%.3f"),
                "Avg_Rev_Util":   st.column_config.NumberColumn("Avg Rev Util",   format="%.3f"),
                "Pct_Delinquent": st.column_config.NumberColumn("% Delinquent",   format="%.1f%%"),
            },
        )
