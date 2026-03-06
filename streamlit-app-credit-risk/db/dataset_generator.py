"""
db/dataset_generator.py
Generates realistic synthetic credit-risk datasets for testing.
"""

import numpy as np
import pandas as pd

COLUMNS = [
    "rev_util", "age", "late_30_59", "debt_ratio", "monthly_inc",
    "open_credit", "late_90", "real_estate", "late_60_89", "dependents",
    "dlq_2yrs",
]


def generate_dataset(n_rows: int, seed: int | None = None) -> pd.DataFrame:
    """Return a DataFrame with *n_rows* of synthetic credit-risk data.

    The distributions are designed to resemble real-world credit data:
      - Most borrowers are low-risk; a minority are high-risk.
      - Late-payment counts are heavily right-skewed (mostly 0).
    """
    rng = np.random.default_rng(seed)

    # ── Continuous features ───────────────────────────────────────────────
    rev_util = np.clip(rng.beta(2, 5, n_rows), 0, 1).round(9)
    age = rng.integers(21, 80, size=n_rows)
    debt_ratio = np.clip(rng.exponential(0.3, n_rows), 0, 5).round(9)
    monthly_inc = np.clip(rng.lognormal(mean=8.5, sigma=0.6, size=n_rows), 1000, 50000).round(0)

    # ── Count features (right-skewed) ─────────────────────────────────────
    late_30_59 = rng.choice([0, 0, 0, 0, 0, 1, 1, 2, 3], size=n_rows)
    late_60_89 = rng.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 2], size=n_rows)
    late_90 = rng.choice([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], size=n_rows)
    open_credit = rng.integers(1, 25, size=n_rows)
    real_estate = rng.choice([0, 0, 1, 1, 1, 2, 2, 3], size=n_rows)
    dependents = rng.choice([0, 0, 0, 1, 1, 2, 2, 3, 4], size=n_rows)

    # ── Target variable (correlated with risk indicators) ─────────────────
    risk_score = (
        0.3 * (rev_util > 0.6).astype(float)
        + 0.2 * (debt_ratio > 0.5).astype(float)
        + 0.15 * (late_30_59 > 0).astype(float)
        + 0.15 * (late_60_89 > 0).astype(float)
        + 0.2 * (late_90 > 0).astype(float)
    )
    prob_default = np.clip(risk_score + rng.normal(0, 0.1, n_rows), 0.02, 0.95)
    dlq_2yrs = (rng.random(n_rows) < prob_default).astype(int)

    # dlq_2yrs is intentionally excluded — the generated file is meant to be
    # run through Batch Prediction first; dlq_2yrs is the model's output.
    df = pd.DataFrame({
        "rev_util": rev_util,
        "age": age,
        "late_30_59": late_30_59,
        "debt_ratio": debt_ratio,
        "monthly_inc": monthly_inc,
        "open_credit": open_credit,
        "late_90": late_90,
        "real_estate": real_estate,
        "late_60_89": late_60_89,
        "dependents": dependents,
    })
    return df
