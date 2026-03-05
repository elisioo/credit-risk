import streamlit as st
from views.main.layout import page_header


def render():
    page_header("Settings", "Configure application and model parameters")
    st.info("🚧  Settings page is under construction. Threshold configuration, alert settings, and integrations will appear here.")
