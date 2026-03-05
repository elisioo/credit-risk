import sys
import os

# Ensure project root is on the path so imports resolve correctly
sys.path.insert(0, os.path.dirname(__file__))

from views.main.layout import page_config
from views.main.sidebar import render_sidebar

# Page modules 
from views.pages import dashboard
from views.pages import risk_analysis
from views.pages import borrower_data
from views.pages import model
from views.pages import predict
from views.pages import report
from views.pages import settings

# Page registry (must match sidebar PAGES keys) 
PAGE_MAP = {
    "Dashboard": dashboard,
    "Risk Analysis": risk_analysis,
    "Borrower Data": borrower_data,
    "Model Performance": model,
    "Predictions": predict,
    "Reports": report,
    "Settings": settings,
}


def main():
    page_config()
    active = render_sidebar()
    module = PAGE_MAP.get(active, dashboard)
    module.render()


if __name__ == "__main__":
    main()
