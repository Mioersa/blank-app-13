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

if not uploaded_files:
    st.warning("👋 Upload at least one CSV to start.")
    st.stop()

# ------------------------------------------------------------
# Load All Files + timestamps
# ------------------------------------------------------------
dfs, upload_times, upload_labels = [], [], []

for uploaded in uploaded_files:
    fn = uploaded.name
    # filename pattern: ***_ddmmyyyy_hhmmss.csv
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
# Dropdown for expiryDate (from first file)
# ------------------------------------------------------------
first_df = dfs[0]
if "expiryDate" not in first_df.columns:
    st.error("❌  Column 'expiryDate' not found in uploaded CSVs.")
    st.stop()

expiry_options = sorted(first_df["expiryDate"].unique())
selected_expiry = st.selectbox("Select Expiry Date (from first file)", expiry_options)

# ------------------------------------------------------------
# Filter all files for selected expiryDate + combine
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
# Display Combined Table for Selected Expiry
# ------------------------------------------------------------
st.subheader(f"🧾 Combined Data for expiryDate = {selected_expiry}")
st.dataframe(final_df)
st.caption(f"{len(final_df):,} rows | {final_df['contract'].nunique()} contracts | {len(filtered_list)} files")

# ------------------------------------------------------------
# New Table: volume summary vs label
# ------------------------------------------------------------
st.subheader("📊 Volume Summary by Capture Time")

# get representative volume per file (e.g. first row by label)
summary_records = []
for lbl in upload_labels:
    sub = final_df[final_df["label"] == lbl]
    if sub.empty:
        continue
    vol_val = sub["volume"].iloc[0] if "volume" in sub else np.nan
    summary_records.append({"time": lbl, "volume": vol_val})

summary_df = pd.DataFrame(summary_records)

# compute volume differences
summary_df["volume_difference"] = summary_df["volume"].diff()

# Display final summary table
st.dataframe(summary_df)

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
st.subheader("📈 Volume & Δ Volume Chart")

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(summary_df["time"], summary_df["volume"],
        marker="o", color="tab:blue", label="Volume")
ax.bar(summary_df["time"], summary_df["volume_difference"],
       color="orange", alpha=0.4, label="Δ Volume")

ax.set_title(f"Volume Change between Captures — Expiry {selected_expiry}")
ax.set_xlabel("Capture Label (Time)")
ax.set_ylabel("Volume / Δ Volume")
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)
