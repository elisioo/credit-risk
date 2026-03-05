import streamlit as st
import os

_BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LOGO = os.path.join(_BASE, "asset", "logo", "brand_logo.png")

PAGES = [
    "Dashboard",
    "Predictions",
    "Borrower Data",
    "Risk Analysis",
    "Model Performance",
    "Reports",
    "Settings",
]



def render_sidebar() -> str:
    """Render sidebar navigation and return the active page name."""
    with st.sidebar:
        st.image(_LOGO, use_container_width=True)
        st.divider()

        if "active_page" not in st.session_state:
            st.session_state.active_page = "Dashboard"

        for page in PAGES:

            is_active = st.session_state.active_page == page
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                f" {page}",
                key=f"nav_{page}",
                type=btn_type,
                use_container_width=True,
            ):
                st.session_state.active_page = page
                st.rerun()

        st.divider()
        st.caption("Francisco | Soroño · v1.1.1")

    return st.session_state.active_page 