import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import matplotlib.pyplot as plt

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
# Load All Files
# ------------------------------------------------------------
dfs, upload_times, upload_labels = [], [], []

for uploaded in uploaded_files:
    fn = uploaded.name
    # extract timestamp from filename pattern ***_ddmmyyyy_hhmmss.csv
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
    st.error("❌ Column 'expiryDate' not found in uploaded CSVs.")
    st.stop()

expiry_options = sorted(first_df["expiryDate"].unique())
selected_expiry = st.selectbox("Select Expiry Date (from first file)", expiry_options)

# ------------------------------------------------------------
# Filter all files for selected expiryDate
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
# Show full combined table
# ------------------------------------------------------------
st.subheader(f"🧾 Combined Data for expiryDate = {selected_expiry}")
st.dataframe(final_df)

# ------------------------------------------------------------
# Build summary table: time(label), volume, lastPrice, Δ volume
# ------------------------------------------------------------
summary_records = []
for lbl in upload_labels:
    sub = final_df[final_df["label"] == lbl]
    if sub.empty:
        continue
    vol_val = sub["volume"].iloc[0] if "volume" in sub else np.nan         # first row volume
    price_val = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan # last row lastPrice
    summary_records.append({
        "time": lbl,
        "volume": vol_val,
        "last_price": price_val,
    })

summary_df = pd.DataFrame(summary_records)
summary_df["Δ Volume"] = summary_df["volume"].diff()

st.subheader("📊 Volume & Last Price Summary")
st.dataframe(summary_df)

# ------------------------------------------------------------
# Plot: Δ Volume (bar) + Last Price (line)
# ------------------------------------------------------------
st.subheader("📈 Last Price vs Δ Volume Chart")

fig, ax1 = plt.subplots(figsize=(10, 4))

# bar plot for Δ Volume
ax1.bar(summary_df["time"], summary_df["Δ Volume"],
        color="orange", alpha=0.5, label="Δ Volume")
ax1.set_ylabel("Δ Volume", color="orange")
ax1.grid(True, axis="y", alpha=0.3)

# twin axis for last price
ax2 = ax1.twinx()
ax2.plot(summary_df["time"], summary_df["last_price"],
         color="tab:blue", marker="o", label="Last Price")
ax2.set_ylabel("Last Price", color="tab:blue")

ax1.set_xlabel("Capture Label (Time)")
ax1.set_title(f"Last Price vs Δ Volume — Expiry {selected_expiry}")

fig.tight_layout()
st.pyplot(fig)
