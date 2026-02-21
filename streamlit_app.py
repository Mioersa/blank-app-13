import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(page_title="Intraday Futures Analytics", layout="wide")

# ------------------------------------------------------------
# Upload Section
# ------------------------------------------------------------
st.sidebar.title("📂 Upload Futures CSVs")
uploaded_files = st.sidebar.file_uploader(
    "Drag‑drop one or more intraday futures CSVs (5‑min interval)",
    type="csv",
    accept_multiple_files=True,
)

# stop early if no files
if not uploaded_files:
    st.warning("👋 Upload at least one CSV to start.")
    st.stop()

# ------------------------------------------------------------
# Load All Files
# ------------------------------------------------------------
dfs, upload_times, upload_labels = [], [], []

for uploaded in uploaded_files:
    fn = uploaded.name
    # extract timestamp from filename ***_ddmmyyyy_hhmmss.csv
    m = re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$", fn)
    base_time = datetime.now()
    label = "unknown"
    if m:
        dd, mm, yyyy, HH, MM, SS = m.groups()
        base_time = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
        label = f"{HH}:{MM}:{SS}"
    upload_times.append(base_time)
    upload_labels.append(label)

    df = pd.read_csv(uploaded)
    df["timestamp"] = base_time + pd.to_timedelta(np.arange(len(df)) * 5, unit="min")
    dfs.append(df)

# ------------------------------------------------------------
# Dropdown for expiryDate from first file
# ------------------------------------------------------------
first_df = dfs[0]
if "expiryDate" not in first_df.columns:
    st.error("❌ Column 'expiryDate' not found in uploaded CSVs.")
    st.stop()

expiry_options = sorted(first_df["expiryDate"].unique())
selected_expiry = st.selectbox("Select Expiry Date (from first file)", expiry_options)

# ------------------------------------------------------------
# Filter all files and merge rows with selected expiryDate
# ------------------------------------------------------------
filtered_list = []
for i, df_file in enumerate(dfs):
    if "expiryDate" not in df_file.columns:
        continue
    sub = df_file[df_file["expiryDate"] == selected_expiry].copy()
    if sub.empty:
        continue
    sub["capture_time"] = upload_times[i]
    sub["label"] = upload_labels[i]
    filtered_list.append(sub)

if not filtered_list:
    st.warning(f"No rows found for expiryDate '{selected_expiry}' in uploaded files.")
    st.stop()

final_df = pd.concat(filtered_list).sort_values(["contract", "timestamp"]).reset_index(drop=True)

# ------------------------------------------------------------
# Display Complete Table
# ------------------------------------------------------------
st.subheader(f"🧾 Combined Data for expiryDate = {selected_expiry}")
st.dataframe(final_df)

# basic info
st.caption(
    f"{len(final_df):,} rows | "
    f"{final_df['contract'].nunique()} contracts | "
    f"Upload captures: {len(filtered_list)} files"
)

# optional summary table
summary = {
    "Total Rows": [len(final_df)],
    "Contracts": [final_df["contract"].nunique()],
    "First Timestamp": [final_df["timestamp"].min()],
    "Last Timestamp": [final_df["timestamp"].max()],
    "Avg Volume": [round(final_df['volume'].mean(), 2) if 'volume' in final_df else np.nan],
    "Avg Close Price": [round(final_df['closePrice'].mean(), 2) if 'closePrice' in final_df else np.nan],
}
st.table(pd.DataFrame(summary))
