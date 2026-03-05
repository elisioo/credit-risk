# Credalytix – Credit Risk Analysis Dashboard

A Streamlit-based web application for credit risk prediction and borrower management, built as a school project.

## Overview

Credalytix provides an interactive dashboard that allows users to analyze credit risk data, manage borrower records, and run real-time default predictions using a trained machine learning model.

## Features

| Page                  | Description                                                                  |
| --------------------- | ---------------------------------------------------------------------------- |
| **Dashboard**         | Overview of key metrics, risk distribution, and model performance trends     |
| **Predictions**       | Single or batch (CSV upload) credit risk prediction using a trained ML model |
| **Borrower Data**     | Browse, search, add, edit, and delete borrower records (SQLite-backed)       |
| **Risk Analysis**     | Visual analysis of risk factors and trends                                   |
| **Model Performance** | Evaluation metrics and performance charts                                    |
| **Reports**           | Generate and view reports                                                    |
| **Settings**          | Application configuration                                                    |

## Tech Stack

- **Frontend:** Streamlit
- **Data:** Pandas, NumPy
- **Visualization:** Plotly
- **Machine Learning:** Scikit-learn
- **Database:** SQLite3

## Project Structure

```
streamlit-app-credit-risk/
├── app.py                  # Entry point
├── requirements.txt
├── asset/logo/             # Brand assets
├── data/
│   ├── Credit_Risk_Benchmark_Dataset.csv
│   └── borrowers.db        # Auto-created on first run
├── db/
│   └── database.py          # SQLite helper (CRUD, pagination)
├── models/
│   └── credit_risk_model.pkl # Trained model
└── views/
    ├── main/
    │   ├── layout.py         # Page config & header
    │   └── sidebar.py        # Navigation
    └── pages/
        ├── dashboard.py
        ├── predict.py
        ├── borrower_data.py
        ├── risk_analysis.py
        ├── model.py
        ├── report.py
        └── settings.py
```

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd streamlit-app-credit-risk

# Install dependencies
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

> On first run, the SQLite database (`data/borrowers.db`) is automatically created and seeded from the CSV dataset.

## ML Model

The prediction page uses a pre-trained Gradient Boosting model (`credit_risk_model.pkl`) with 10 input features:

| Feature       | Description                              |
| ------------- | ---------------------------------------- |
| `rev_util`    | Revolving credit utilization rate (0–1)  |
| `age`         | Borrower's age                           |
| `late_30_59`  | Times 30-59 days past due (last 2 years) |
| `debt_ratio`  | Monthly debt payments / gross income     |
| `monthly_inc` | Gross monthly income                     |
| `open_credit` | Number of open credit lines              |
| `late_90`     | Times 90+ days past due (last 2 years)   |
| `real_estate` | Number of real estate loans              |
| `late_60_89`  | Times 60-89 days past due (last 2 years) |
| `dependents`  | Number of dependents                     |

## Creators

**Francisco, Jan-Rel | Soroño, Eli**

## Disclaimer

This project is for **educational purposes only** and is not intended for production use or real financial decision-making.
