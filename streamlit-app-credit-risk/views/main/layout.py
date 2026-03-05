import streamlit as st


def page_config():
    """Call once at the very top of app.py."""
    st.set_page_config(
        page_title="CreditAI – Credit Risk Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Right-align KPI metric values — no native param exists for this
    st.markdown(
        """
        <style>
            [data-testid="stMetricValue"] { text-align: right; }
            [data-testid="stMetricDelta"] { justify-content: flex-end; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = ""):
    """Render a consistent page header."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)