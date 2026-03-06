"""
db/predictions_history.py
Persistent log of every prediction made in the app.

Each record captures:
  - All feature inputs
  - dlq_2yrs (predicted value)
  - default_prob_% (probability)
  - prediction label (DEFAULT / NO DEFAULT)
  - risk_level
  - predicted_at timestamp
  - source  ('single' or 'batch')
"""

import os
import sqlite3
import pandas as pd

_BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_BASE, "data", "predictions_history.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_table() -> None:
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                rev_util        REAL,
                age             INTEGER,
                late_30_59      INTEGER,
                debt_ratio      REAL,
                monthly_inc     REAL,
                open_credit     INTEGER,
                late_90         INTEGER,
                real_estate     INTEGER,
                late_60_89      INTEGER,
                dependents      INTEGER,
                dlq_2yrs        INTEGER,
                default_prob    REAL,
                prediction      TEXT,
                risk_level      TEXT,
                source          TEXT DEFAULT 'single',
                predicted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def log_single(
    rev_util: float, age: int, late_30_59: int, debt_ratio: float,
    monthly_inc: float, open_credit: int, late_90: int, real_estate: int,
    late_60_89: int, dependents: int,
    dlq_2yrs: int, default_prob: float,
) -> None:
    init_table()
    prediction = "DEFAULT" if dlq_2yrs == 1 else "NO DEFAULT"
    if default_prob < 30:
        risk = "Low"
    elif default_prob < 60:
        risk = "Moderate"
    else:
        risk = "High"

    with _conn() as con:
        con.execute("""
            INSERT INTO history
            (rev_util, age, late_30_59, debt_ratio, monthly_inc, open_credit,
             late_90, real_estate, late_60_89, dependents,
             dlq_2yrs, default_prob, prediction, risk_level, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'single')
        """, (rev_util, age, late_30_59, debt_ratio, monthly_inc, open_credit,
              late_90, real_estate, late_60_89, dependents,
              dlq_2yrs, round(default_prob, 2), prediction, risk))


def log_batch(results_df: pd.DataFrame) -> int:
    """Bulk-log all rows from a batch prediction results DataFrame."""
    init_table()
    df = results_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    rows = []
    for _, r in df.iterrows():
        prob = float(r.get("default_prob_%", 0))
        rows.append((
            float(r.get("rev_util", 0)),
            int(r.get("age", 0)),
            int(r.get("late_30_59", 0)),
            float(r.get("debt_ratio", 0)),
            float(r.get("monthly_inc", 0)),
            int(r.get("open_credit", 0)),
            int(r.get("late_90", 0)),
            int(r.get("real_estate", 0)),
            int(r.get("late_60_89", 0)),
            int(r.get("dependents", 0)),
            int(r.get("dlq_2yrs", 0)),
            round(prob, 2),
            str(r.get("prediction", "")),
            str(r.get("risk_level", "")),
        ))

    with _conn() as con:
        con.executemany("""
            INSERT INTO history
            (rev_util, age, late_30_59, debt_ratio, monthly_inc, open_credit,
             late_90, real_estate, late_60_89, dependents,
             dlq_2yrs, default_prob, prediction, risk_level, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'batch')
        """, rows)
    return len(rows)


def fetch_history(limit: int = 500) -> pd.DataFrame:
    init_table()
    with _conn() as con:
        rows = con.execute("""
            SELECT id, predicted_at, source, prediction, risk_level,
                   default_prob, dlq_2yrs, rev_util, age, late_30_59,
                   debt_ratio, monthly_inc, open_credit, late_90,
                   real_estate, late_60_89, dependents
            FROM history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def total_count() -> int:
    init_table()
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM history").fetchone()[0]


def clear_history() -> None:
    init_table()
    with _conn() as con:
        con.execute("DELETE FROM history")
