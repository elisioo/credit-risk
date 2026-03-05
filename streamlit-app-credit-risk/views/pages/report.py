import streamlit as st
from views.main.layout import page_header


def render():
    page_header("Reports", "Generate and download risk reports")
    st.info("🚧  Reports page is under construction. Scheduled and ad-hoc reports will be available here.")
