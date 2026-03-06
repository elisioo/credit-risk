import io
import streamlit as st
from views.main.layout import page_header
from db.database import (
    init_db, fetch_stats, fetch_all,
    get_settings, save_settings,
    rebuild_risk_levels, reset_db,
    DB_PATH, SETTINGS_PATH, _DEFAULTS,
)


def render():
    init_db()
    page_header("Settings", "Configure application preferences and data management")

    cfg = get_settings()

    # ── Risk Scoring Rules ────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Risk Scoring Rules")
        st.caption(
            "These thresholds control how borrowers are classified as **Low**, **Medium**, "
            "or **High** risk when records are added or updated. Changes apply to new and "
            "edited records immediately; click **Rebuild** to reclassify all existing records."
        )

        c1, c2 = st.columns(2)
        new_rev  = c1.slider(
            "Revolving Utilization threshold (Medium risk)",
            min_value=0.10, max_value=1.00, step=0.05,
            value=float(cfg["rev_util_threshold"]),
            help="Borrowers with rev_util above this value are classified as at least Medium risk",
            format="%.2f",
        )
        new_debt = c2.slider(
            "Debt Ratio threshold (Medium risk)",
            min_value=0.10, max_value=1.00, step=0.05,
            value=float(cfg["debt_ratio_threshold"]),
            help="Borrowers with debt_ratio above this value are classified as at least Medium risk",
            format="%.2f",
        )

        st.caption(
            "**Classification logic:** `dlq_2yrs = 1` → **High** · "
            f"`rev_util > {new_rev:.2f}` or `debt_ratio > {new_debt:.2f}` → **Medium** · "
            "otherwise → **Low**"
        )

        sa, rb = st.columns([1, 1])
        if sa.button("Save Thresholds", type="primary", use_container_width=True):
            cfg["rev_util_threshold"]   = new_rev
            cfg["debt_ratio_threshold"] = new_debt
            save_settings(cfg)
            st.success("Thresholds saved. New/edited records will use the updated rules.")

        if rb.button("Rebuild All Risk Levels", use_container_width=True):
            cfg["rev_util_threshold"]   = new_rev
            cfg["debt_ratio_threshold"] = new_debt
            save_settings(cfg)
            n = rebuild_risk_levels()
            st.success(f"Risk levels rebuilt for {n:,} borrower records.")
            st.cache_data.clear()

    st.write("")

    # ── Display Preferences ───────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Display Preferences")
        new_page_size = st.select_slider(
            "Rows per page (Borrower Data table)",
            options=[10, 25, 50, 100, 200],
            value=int(cfg["page_size"]),
        )
        if st.button("Save Preferences", type="primary"):
            cfg["page_size"] = new_page_size
            save_settings(cfg)
            # patch the module-level constant so fetch_page picks it up
            import db.database as _dbmod
            _dbmod.PAGE_SIZE = new_page_size
            st.success(f"Page size updated to {new_page_size} rows.")

    st.write("")

    # ── Data Management ───────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("Data Management")

        stats = fetch_stats()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Records",   f"{stats['total']:,}")
        m2.metric("High Risk",        f"{stats['high']:,}")
        m3.metric("Medium Risk",      f"{stats['medium']:,}")
        m4.metric("Low Risk",         f"{stats['low']:,}")

        st.write("")

        # Export CSV
        df = fetch_all()
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export All Borrower Data (.csv)",
                data=csv_bytes,
                file_name="borrowers_export.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.write("")

        # Reset database — guarded by a confirmation expander
        with st.expander("Danger Zone — Reset Database", expanded=False):
            st.warning(
                "This will **permanently delete** the database file and reseed it from the "
                "original CSV on the next page load. All manually added or edited records "
                "will be lost and cannot be recovered."
            )
            confirm = st.text_input(
                "Type **RESET** to confirm",
                placeholder="RESET",
                label_visibility="collapsed",
            )
            if st.button("Reset Database", type="primary", use_container_width=True):
                if confirm.strip() == "RESET":
                    reset_db()
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Database deleted. It will be reseeded on the next page load.")
                else:
                    st.error("Please type RESET exactly to confirm.")

    st.write("")