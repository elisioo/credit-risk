import streamlit as st
import pandas as pd
from views.main.layout import page_header
from db.database import (
    init_db, fetch_page, fetch_one,
    add_borrower, update_borrower, delete_borrower,
    PAGE_SIZE,
)

RISK_BADGE   = {"High": "High", "Medium": "Medium", "Low": "Low"}
RISK_OPTIONS = ["Low", "Medium", "High"]


def _state(key, default=None):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def render():
    init_db()  # no-op after first run
    page_header("Borrower Data", "Browse and manage individual borrower records")

    # ── Toolbar ───────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([5, 3])

    with left_col:
        s_sub, rf_sub = st.columns([2, 1])
        search      = s_sub.text_input("Search", placeholder="ID / Age / Risk Level…", label_visibility="collapsed")
        risk_filter = rf_sub.selectbox("Risk", ["All", "Low", "Medium", "High"], label_visibility="collapsed")

    with right_col:
        sp, add_col, edit_col, del_col = st.columns([1, 1, 1, 1])
        sp.write("")  # spacer to push buttons right
        add_clicked  = add_col.button("Add",    type="primary", use_container_width=True)
        edit_clicked = edit_col.button("Edit",   use_container_width=True)
        del_clicked  = del_col.button("Delete",  use_container_width=True)

    # ── Pagination state ──────────────────────────────────────────────────────
    _state("bdata_page", 1)
    # Reset to page 1 when search/filter changes
    prev_search = _state("bdata_prev_search", "")
    prev_filter = _state("bdata_prev_filter", "All")
    if search != prev_search or risk_filter != prev_filter:
        st.session_state["bdata_page"]        = 1
        st.session_state["bdata_prev_search"] = search
        st.session_state["bdata_prev_filter"] = risk_filter

    current_page = st.session_state["bdata_page"]

    # ── Fetch from SQLite ─────────────────────────────────────────────────────
    df, total = fetch_page(current_page, search, risk_filter)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    # ── Table ─────────────────────────────────────────────────────────────────
    with st.container(border=True):
        # Pagination controls inside the box
        pg_info, prev_btn, pg_num, next_btn = st.columns([4, 1, 1, 1])
        pg_info.caption(f"Showing {min((current_page-1)*PAGE_SIZE+1, total)}–{min(current_page*PAGE_SIZE, total)} of **{total:,}** records  ·  Page {current_page}/{total_pages}")
        if prev_btn.button("◀", disabled=current_page <= 1):
            st.session_state["bdata_page"] -= 1
            st.rerun()
        pg_num.write(f"**{current_page}**")
        if next_btn.button("▶", disabled=current_page >= total_pages):
            st.session_state["bdata_page"] += 1
            st.rerun()

        if df.empty:
            st.info("No records match your search.")
            return

        display_df = df.copy()
        display_df["risk_level"] = display_df["risk_level"].map(RISK_BADGE)

        selection = st.dataframe(
            display_df,
            hide_index=True,
            width="stretch",
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "id":           st.column_config.NumberColumn("ID",            width="small",  format="%d"),
                "age":          st.column_config.NumberColumn("Age",           width="small",  format="%d"),
                "rev_util":     st.column_config.NumberColumn("Rev. Util",     width="small",  format="%.3f"),
                "debt_ratio":   st.column_config.NumberColumn("Debt Ratio",    width="small",  format="%.3f"),
                "monthly_inc":  st.column_config.NumberColumn("Monthly Inc",   width="medium", format="%.0f"),
                "open_credit":  st.column_config.NumberColumn("Open Credit",   width="small",  format="%d"),
                "late_90":      st.column_config.NumberColumn("Late 90+",      width="small",  format="%d"),
                "late_30_59":   st.column_config.NumberColumn("Late 30-59",    width="small",  format="%d"),
                "late_60_89":   st.column_config.NumberColumn("Late 60-89",    width="small",  format="%d"),
                "real_estate":  st.column_config.NumberColumn("Real Estate",   width="small",  format="%d"),
                "dependents":   st.column_config.NumberColumn("Dependents",    width="small",  format="%d"),
                "dlq_2yrs":     st.column_config.NumberColumn("Delinquent",    width="small",  format="%d"),
                "risk_level":   st.column_config.TextColumn("Risk Level",      width="small"),
            },
        )

    # Resolve selected row → borrower ID
    selected_rows = selection.selection.get("rows", [])
    selected_id   = int(df.iloc[selected_rows[0]]["id"]) if selected_rows else None

    # ── ADD ───────────────────────────────────────────────────────────────────
    if add_clicked:
        st.session_state["crud_mode"] = "add"

    if st.session_state.get("crud_mode") == "add":
        with st.form("form_add", border=True):
            st.subheader("Add Borrower")
            c1, c2, c3 = st.columns(3)
            age         = c1.number_input("Age",               min_value=18, max_value=100, value=35)
            monthly_inc = c2.number_input("Monthly Income",    min_value=0.0, value=5000.0)
            rev_util    = c3.number_input("Revolving Util.",   min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.3f")
            debt_ratio  = c1.number_input("Debt Ratio",        min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.3f")
            open_credit = c2.number_input("Open Credit Lines", min_value=0, value=5)
            real_estate = c3.number_input("Real Estate Loans", min_value=0, value=1)
            late_30_59  = c1.number_input("Late 30-59 days",   min_value=0, value=0)
            late_60_89  = c2.number_input("Late 60-89 days",   min_value=0, value=0)
            late_90     = c3.number_input("Late 90+ days",     min_value=0, value=0)
            dependents  = c1.number_input("Dependents",        min_value=0, value=0)
            dlq_2yrs    = c2.selectbox("Delinquent 2yrs",      [0, 1])

            sub, cancel = st.columns(2)
            if sub.form_submit_button("Save", type="primary"):
                new_id = add_borrower(
                    age=int(age), rev_util=float(rev_util), debt_ratio=float(debt_ratio),
                    monthly_inc=float(monthly_inc), open_credit=int(open_credit),
                    late_90=int(late_90), dlq_2yrs=int(dlq_2yrs),
                    late_30_59=int(late_30_59), late_60_89=int(late_60_89),
                    real_estate=int(real_estate), dependents=int(dependents),
                )
                st.session_state["crud_mode"] = None
                st.success(f"Borrower #{new_id} added.")
                st.rerun()
            if cancel.form_submit_button("Cancel"):
                st.session_state["crud_mode"] = None
                st.rerun()

    # ── EDIT ──────────────────────────────────────────────────────────────────
    if edit_clicked:
        if selected_id:
            st.session_state["crud_mode"] = "edit"
        else:
            st.warning("Select a row to edit.")

    if st.session_state.get("crud_mode") == "edit" and selected_id:
        row = fetch_one(selected_id)
        if row:
            with st.form("form_edit", border=True):
                st.subheader(f"Edit Borrower #{selected_id}")
                c1, c2, c3 = st.columns(3)
                age         = c1.number_input("Age",               min_value=18, max_value=100, value=int(row["age"]))
                monthly_inc = c2.number_input("Monthly Income",    min_value=0.0, value=float(row["monthly_inc"]))
                rev_util    = c3.number_input("Revolving Util.",   min_value=0.0, max_value=1.0, value=float(row["rev_util"]), step=0.01, format="%.3f")
                debt_ratio  = c1.number_input("Debt Ratio",        min_value=0.0, max_value=1.0, value=float(row["debt_ratio"]), step=0.01, format="%.3f")
                open_credit = c2.number_input("Open Credit Lines", min_value=0, value=int(row["open_credit"]))
                real_estate = c3.number_input("Real Estate Loans", min_value=0, value=int(row["real_estate"]))
                late_30_59  = c1.number_input("Late 30-59 days",   min_value=0, value=int(row["late_30_59"]))
                late_60_89  = c2.number_input("Late 60-89 days",   min_value=0, value=int(row["late_60_89"]))
                late_90     = c3.number_input("Late 90+ days",     min_value=0, value=int(row["late_90"]))
                dependents  = c1.number_input("Dependents",        min_value=0, value=int(row["dependents"]))
                dlq_2yrs    = c2.selectbox("Delinquent 2yrs",      [0, 1], index=int(row["dlq_2yrs"]))

                sub, cancel = st.columns(2)
                if sub.form_submit_button("Update", type="primary"):
                    update_borrower(
                        borrower_id=selected_id,
                        age=int(age), rev_util=float(rev_util), debt_ratio=float(debt_ratio),
                        monthly_inc=float(monthly_inc), open_credit=int(open_credit),
                        late_90=int(late_90), dlq_2yrs=int(dlq_2yrs),
                        late_30_59=int(late_30_59), late_60_89=int(late_60_89),
                        real_estate=int(real_estate), dependents=int(dependents),
                    )
                    st.session_state["crud_mode"] = None
                    st.success(f"Borrower #{selected_id} updated.")
                    st.rerun()
                if cancel.form_submit_button("Cancel"):
                    st.session_state["crud_mode"] = None
                    st.rerun()

    # ── DELETE ────────────────────────────────────────────────────────────────
    if del_clicked:
        if selected_id:
            st.session_state["crud_mode"] = "delete"
        else:
            st.warning("Select a row to delete.")

    if st.session_state.get("crud_mode") == "delete" and selected_id:
        with st.container(border=True):
            st.warning(f"Delete borrower **#{selected_id}**? This cannot be undone.")
            c1, c2 = st.columns(2)
            if c1.button("Yes, delete", type="primary"):
                delete_borrower(selected_id)
                st.session_state["crud_mode"] = None
                st.success(f"Borrower #{selected_id} deleted.")
                st.rerun()
            if c2.button("Cancel"):
                st.session_state["crud_mode"] = None
                st.rerun()
