"""
db/uploaded_datasets.py
Separate storage for user-uploaded test datasets.

Uses a dedicated SQLite database (data/uploaded_datasets.db) so it never
interferes with the main borrowers database.
"""

import os
import sqlite3
import pandas as pd

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_BASE, "data", "uploaded_datasets.db")

_COLS = [
    "rev_util", "age", "late_30_59", "debt_ratio", "monthly_inc",
    "open_credit", "late_90", "real_estate", "late_60_89", "dependents",
    "dlq_2yrs",
]


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_table() -> None:
    """Create the uploaded_data table if it does not exist."""
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_data (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                rev_util     REAL,
                age          INTEGER,
                late_30_59   INTEGER,
                debt_ratio   REAL,
                monthly_inc  REAL,
                open_credit  INTEGER,
                late_90      INTEGER,
                real_estate  INTEGER,
                late_60_89   INTEGER,
                dependents   INTEGER,
                dlq_2yrs     INTEGER,
                uploaded_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def insert_dataframe(df: pd.DataFrame) -> int:
    """Insert all rows from *df* into the uploaded_data table.
    Returns the number of rows inserted."""
    init_table()
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Keep only recognised columns, fill missing with 0
    for c in _COLS:
        if c not in df.columns:
            df[c] = 0
    df = df[_COLS].fillna(0)

    rows = []
    for _, r in df.iterrows():
        rows.append((
            float(r["rev_util"]),
            int(r["age"]),
            int(r["late_30_59"]),
            float(r["debt_ratio"]),
            float(r["monthly_inc"]),
            int(r["open_credit"]),
            int(r["late_90"]),
            int(r["real_estate"]),
            int(r["late_60_89"]),
            int(r["dependents"]),
            int(r["dlq_2yrs"]),
        ))

    with _conn() as con:
        con.executemany("""
            INSERT INTO uploaded_data
            (rev_util, age, late_30_59, debt_ratio, monthly_inc,
             open_credit, late_90, real_estate, late_60_89,
             dependents, dlq_2yrs)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, rows)
    return len(rows)


def fetch_all() -> pd.DataFrame:
    """Return every row from uploaded_data."""
    init_table()
    with _conn() as con:
        rows = con.execute(
            """
            SELECT id, rev_util, age, late_30_59, debt_ratio, monthly_inc,
                   open_credit, late_90, real_estate, late_60_89,
                   dependents, dlq_2yrs, uploaded_at
            FROM uploaded_data
            ORDER BY id DESC
            """
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


def row_count() -> int:
    init_table()
    with _conn() as con:
        return con.execute("SELECT COUNT(*) FROM uploaded_data").fetchone()[0]


def clear_all() -> None:
    """Delete all uploaded data."""
    init_table()
    with _conn() as con:
        con.execute("DELETE FROM uploaded_data")
