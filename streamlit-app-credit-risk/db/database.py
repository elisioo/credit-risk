"""
db/database.py
SQLite helper for the Borrower Data page.

Database file: data/borrowers.db
Source CSV:    data/Credit_Risk_Benchmark_Dataset.csv

The 'borrowers' table is seeded from the CSV on first run (when the DB file
does not exist).  All CRUD operations go through this module so the page
code stays clean.
"""

import os
import sqlite3
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
DB_PATH  = os.path.join(_BASE, "data", "borrowers.db")
CSV_PATH = os.path.join(_BASE, "data", "Credit_Risk_Benchmark_Dataset.csv")

PAGE_SIZE = 50  # rows per page


# ── Internal helpers ──────────────────────────────────────────────────────────

def _conn() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _derive_risk(rev_util: float, debt_ratio: float, dlq: int) -> str:
    if dlq == 1:
        return "High"
    if rev_util > 0.60 or debt_ratio > 0.50:
        return "Medium"
    return "Low"


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create & seed the DB from CSV if it doesn't exist yet."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS borrowers (
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
                risk_level   TEXT
            )
        """)

        # Seed only if the table is empty
        count = con.execute("SELECT COUNT(*) FROM borrowers").fetchone()[0]
        if count == 0 and os.path.exists(CSV_PATH):
            df = pd.read_csv(CSV_PATH)
            df.columns = [c.lower().strip() for c in df.columns]
            df = df.fillna(0)

            rows = []
            for _, r in df.iterrows():
                risk = _derive_risk(
                    float(r.get("rev_util", 0)),
                    float(r.get("debt_ratio", 0)),
                    int(r.get("dlq_2yrs", 0)),
                )
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
                    risk,
                ))

            con.executemany("""
                INSERT INTO borrowers
                (rev_util, age, late_30_59, debt_ratio, monthly_inc,
                 open_credit, late_90, real_estate, late_60_89,
                 dependents, dlq_2yrs, risk_level)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)


# ── Read (paginated + search) ─────────────────────────────────────────────────

def fetch_page(
    page: int = 1,
    search: str = "",
    risk_filter: str = "All",
) -> tuple[pd.DataFrame, int]:
    """
    Returns (DataFrame for this page, total_matching_rows).
    page is 1-indexed.
    """
    offset = (page - 1) * PAGE_SIZE

    where_clauses = []
    params: list = []

    if search.strip():
        like = f"%{search.strip()}%"
        where_clauses.append(
            "(CAST(id AS TEXT) LIKE ? "
            "OR CAST(age AS TEXT) LIKE ? "
            "OR risk_level LIKE ?)"
        )
        params += [like, like, like]

    if risk_filter != "All":
        where_clauses.append("risk_level = ?")
        params.append(risk_filter)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    with _conn() as con:
        total = con.execute(
            f"SELECT COUNT(*) FROM borrowers {where_sql}", params
        ).fetchone()[0]

        rows = con.execute(
            f"""
            SELECT id, age, rev_util, debt_ratio, monthly_inc,
                   open_credit, late_90, dlq_2yrs, risk_level,
                   late_30_59, late_60_89, real_estate, dependents
            FROM borrowers {where_sql}
            ORDER BY id
            LIMIT ? OFFSET ?
            """,
            params + [PAGE_SIZE, offset],
        ).fetchall()

    df = pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()
    return df, total


# ── Stats for dashboard ───────────────────────────────────────────────────────

def fetch_stats() -> dict:
    with _conn() as con:
        total     = con.execute("SELECT COUNT(*) FROM borrowers").fetchone()[0]
        high      = con.execute("SELECT COUNT(*) FROM borrowers WHERE risk_level='High'").fetchone()[0]
        med       = con.execute("SELECT COUNT(*) FROM borrowers WHERE risk_level='Medium'").fetchone()[0]
        low       = con.execute("SELECT COUNT(*) FROM borrowers WHERE risk_level='Low'").fetchone()[0]
        avg_inc   = con.execute("SELECT AVG(monthly_inc) FROM borrowers").fetchone()[0] or 0
        avg_debt  = con.execute("SELECT AVG(debt_ratio) FROM borrowers").fetchone()[0] or 0
        avg_rev   = con.execute("SELECT AVG(rev_util) FROM borrowers").fetchone()[0] or 0
    return {
        "total": total, "high": high, "medium": med, "low": low,
        "avg_monthly_inc": avg_inc, "avg_debt_ratio": avg_debt,
        "avg_rev_util": avg_rev,
    }


def fetch_recent(limit: int = 5) -> pd.DataFrame:
    """Return the most recently added borrowers."""
    with _conn() as con:
        rows = con.execute(
            """
            SELECT id, age, rev_util, debt_ratio, monthly_inc,
                   open_credit, late_90, dlq_2yrs, risk_level,
                   late_30_59, late_60_89, real_estate, dependents
            FROM borrowers
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return pd.DataFrame([dict(r) for r in rows]) if rows else pd.DataFrame()


# ── Create ────────────────────────────────────────────────────────────────────

def add_borrower(
    age: int, rev_util: float, debt_ratio: float, monthly_inc: float,
    open_credit: int, late_90: int, dlq_2yrs: int,
    late_30_59: int = 0, late_60_89: int = 0,
    real_estate: int = 0, dependents: int = 0,
) -> int:
    risk = _derive_risk(rev_util, debt_ratio, dlq_2yrs)
    with _conn() as con:
        cur = con.execute("""
            INSERT INTO borrowers
            (rev_util, age, late_30_59, debt_ratio, monthly_inc,
             open_credit, late_90, real_estate, late_60_89,
             dependents, dlq_2yrs, risk_level)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (rev_util, age, late_30_59, debt_ratio, monthly_inc,
              open_credit, late_90, real_estate, late_60_89,
              dependents, dlq_2yrs, risk))
        return cur.lastrowid


# ── Update ────────────────────────────────────────────────────────────────────

def update_borrower(
    borrower_id: int,
    age: int, rev_util: float, debt_ratio: float, monthly_inc: float,
    open_credit: int, late_90: int, dlq_2yrs: int,
    late_30_59: int = 0, late_60_89: int = 0,
    real_estate: int = 0, dependents: int = 0,
) -> None:
    risk = _derive_risk(rev_util, debt_ratio, dlq_2yrs)
    with _conn() as con:
        con.execute("""
            UPDATE borrowers SET
                rev_util=?, age=?, late_30_59=?, debt_ratio=?,
                monthly_inc=?, open_credit=?, late_90=?,
                real_estate=?, late_60_89=?, dependents=?,
                dlq_2yrs=?, risk_level=?
            WHERE id=?
        """, (rev_util, age, late_30_59, debt_ratio, monthly_inc,
              open_credit, late_90, real_estate, late_60_89,
              dependents, dlq_2yrs, risk, borrower_id))


# ── Delete ────────────────────────────────────────────────────────────────────

def delete_borrower(borrower_id: int) -> None:
    with _conn() as con:
        con.execute("DELETE FROM borrowers WHERE id=?", (borrower_id,))


# ── Fetch single row ──────────────────────────────────────────────────────────

def fetch_one(borrower_id: int) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM borrowers WHERE id=?", (borrower_id,)
        ).fetchone()
    return dict(row) if row else None
