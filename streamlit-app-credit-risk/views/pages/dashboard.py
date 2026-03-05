import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from views.main.layout import page_header

# ── Sample / placeholder data ─────────────────────────────────────────────────

RISK_PIE_DATA = {
    "labels": ["Low Risk", "Medium Risk", "High Risk"],
    "values": [21_426, 15_234, 3_421],
    "colors": ["#10B981", "#F59E0B", "#EF4444"],
}

PERFORMANCE_TREND = {
    "months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "accuracy": [84.2, 85.1, 86.3, 85.8, 86.9, 87.3],
}

RECENT_PREDICTIONS = pd.DataFrame(
    {
        "Borrower ID": ["BR-001847", "BR-002156", "BR-003298", "BR-004511", "BR-005673"],
        "Risk Score": [742, 681, 598, 720, 655],
        "Default Probability": ["23.4%", "15.8%", "8.2%", "19.6%", "11.3%"],
        "Risk Level": ["High", "Medium", "Low", "High", "Medium"],
        "Assessment Date": [
            "2026-03-01",
            "2026-03-02",
            "2026-03-03",
            "2026-03-04",
            "2026-03-05",
        ],
    }
)

RISK_BADGE = {
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
}


# ── Chart builders ─────────────────────────────────────────────────────────────


def _kpi_card(
    container,
    label: str,
    value: str,
    delta: str,
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


def _risk_pie() -> go.Figure:
    fig = go.Figure(
        go.Pie(
            labels=RISK_PIE_DATA["labels"],
            values=RISK_PIE_DATA["values"],
            marker_colors=RISK_PIE_DATA["colors"],
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


def _performance_trend() -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=PERFORMANCE_TREND["months"],
            y=PERFORMANCE_TREND["accuracy"],
            mode="lines+markers",
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=7, color="#3B82F6"),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.08)",
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Month",
        yaxis=dict(title="Accuracy %", range=[80, 90]),
        margin=dict(t=10, b=50, l=50, r=20),
        height=300,
    )
    return fig


# ── Main render function ───────────────────────────────────────────────────────


def render():
    page_header(
        "Dashboard",
        "Monitor borrower risk and model performance in real-time",
    )

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    _kpi_card(kpi1, "Total Borrowers",        "24,847", "+2.3% from last month",  help="Total active borrowers in portfolio")
    _kpi_card(kpi2, "High Risk Borrowers",     "3,421",  "+0.8% from last month",  delta_color="inverse", help="Borrowers with risk score > threshold")
    _kpi_card(kpi3, "Avg Default Probability", "13.8%",  "-1.2% from last month",  help="Mean predicted default probability")
    _kpi_card(kpi4, "Model Accuracy",          "87.3%",  "+0.5% from last month",  help="Current model accuracy on validation set")

    # ── Charts ─────────────────────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.subheader("Risk Distribution")
            st.plotly_chart(_risk_pie(), config={"displayModeBar": False})

    with chart_col2:
        with st.container(border=True):
            st.subheader("Model Performance Trend")
            st.plotly_chart(_performance_trend(), config={"displayModeBar": False})

    # ── Recent Predictions + Quick Actions ────────────────────────────────────
    table_col, actions_col = st.columns([2, 1])

    with table_col:
        with st.container(border=True):
            st.subheader("Recent Predictions")

            display_df = RECENT_PREDICTIONS.copy()
            display_df["Risk Level"] = display_df["Risk Level"].map(RISK_BADGE)

            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Borrower ID": st.column_config.TextColumn(
                        "Borrower ID", width="small"
                    ),
                    "Risk Score": st.column_config.NumberColumn(
                        "Risk Score", width="small", format="%d"
                    ),
                    "Default Probability": st.column_config.TextColumn(
                        "Default Prob.", width="small"
                    ),
                    "Risk Level": st.column_config.TextColumn(
                        "Risk Level", width="small"
                    ),
                    "Assessment Date": st.column_config.DateColumn(
                        "Date", width="medium", format="MMM DD, YYYY"
                    ),
                },
            )
