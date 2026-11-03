# =========================================
# pages/6_Monitoring.py — Data Drift Monitoring
# =========================================

import pandas as pd
import streamlit as st

from src.monitoring import (
    ALERT_DRIFTED_COLUMNS,
    MIN_RECORDS,
    RECENT_RECORDS,
    check_drift,
    load_reference_and_holdout,
    simulate_drift,
)
from src.prediction_log import load_predictions
from src.ui import apply_page_style

apply_page_style()

LOGGED = "Logged predictions"
HEALTHY = "Held-out test set (healthy baseline)"
SIMULATED = "Simulated drift (demo)"

# -----------------------------------------
# PAGE HEADER
# -----------------------------------------
st.markdown("""
<h2 style="margin-bottom:4px;">📡 Data Drift Monitoring</h2>
<p style="color:#64748b; margin-top:0;">
Are recent customers still similar to the data the model was trained on?
</p>
""", unsafe_allow_html=True)


# -----------------------------------------
# DRIFT CHECK (CACHED)
# -----------------------------------------
@st.cache_data(ttl=60)
def run_check(source):
    """Returns (summary, report_html), or (None, None) when there is too little data to judge."""
    if source == LOGGED:
        current = load_predictions(limit=RECENT_RECORDS)
        if len(current) < MIN_RECORDS:
            return None, None
        reference, _ = load_reference_and_holdout()
    else:
        reference, holdout = load_reference_and_holdout()
        current = holdout if source == HEALTHY else simulate_drift(holdout)

    summary, snapshot = check_drift(current, reference)
    return summary, snapshot.get_html_str(as_iframe=False)


source = st.radio("Data to check against the training data", [LOGGED, HEALTHY, SIMULATED], horizontal=True)

if source == SIMULATED:
    st.caption("Demo data: the held-out test set with handset age and minutes shifted on purpose. This is not real traffic.")

with st.spinner("Checking for drift..."):
    summary, report_html = run_check(source)

if summary is None:
    st.info(f"Needs at least {MIN_RECORDS} logged API predictions to give a reliable result. Fewer rows cause false alarms.")
    st.stop()

# -----------------------------------------
# RESULT
# -----------------------------------------
drifted = summary["drifted_columns"]

if summary["alert"]:
    st.error(f"🚨 Drift alert: {len(drifted)} columns drifted (alert at {ALERT_DRIFTED_COLUMNS}): {', '.join(drifted)}")
else:
    st.success(f"✅ No drift alert: {len(drifted)} columns drifted (alert at {ALERT_DRIFTED_COLUMNS}).")

col1, col2 = st.columns(2)
col1.metric("Records Checked", f"{summary['records']:,}")
col2.metric("Drifted Columns", f"{len(drifted)} of {len(summary['columns'])}")

st.dataframe(pd.DataFrame(summary["columns"]), use_container_width=True, hide_index=True)

st.download_button("Download full Evidently report", report_html, file_name="evidently_report.html", mime="text/html")
