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
# Show combined data table
# ------------------------------------------------------------
st.subheader(f"🧾 Combined Data for expiryDate = {selected_expiry}")
st.dataframe(final_df)

# ------------------------------------------------------------
# Build summary table: label/time, volume, lastPrice, Δ Volume
# ------------------------------------------------------------
summary_records = []
for lbl in upload_labels:
    sub = final_df[final_df["label"] == lbl]
    if sub.empty:
        continue
    vol_val = sub["volume"].iloc[0] if "volume" in sub else np.nan
    price_val = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
    summary_records.append({"time": lbl, "volume": vol_val, "last_price": price_val})

summary_df = pd.DataFrame(summary_records)
summary_df["Δ Volume"] = summary_df["volume"].diff()

st.subheader("📊 Volume & Last Price Summary")
st.dataframe(summary_df)

# ------------------------------------------------------------
# Split Chart: Top = Last Price line, Bottom = Δ Volume bars (30%)
# ------------------------------------------------------------
st.subheader("📈 Last Price (top) & Δ Volume (bottom 30%) Chart")

fig, (ax_price, ax_vol) = plt.subplots(
    2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [0.7, 0.3]}, sharex=True
)

# --- top: last price ---
ax_price.plot(summary_df["time"], summary_df["last_price"],
              color="tab:blue", marker="o", label="Last Price")
ax_price.set_title(f"Last Price and Δ Volume — Expiry {selected_expiry}")
ax_price.set_ylabel("Last Price")
ax_price.legend(loc="upper left")
ax_price.grid(True, alpha=0.3)

# --- bottom: Δ Volume bars ---
ax_vol.bar(summary_df["time"], summary_df["Δ Volume"],
           color="orange", alpha=0.6, label="Δ Volume")
ax_vol.axhline(0, color="gray", linewidth=0.8)
ax_vol.set_ylabel("Δ Volume")
ax_vol.legend(loc="upper left")
ax_vol.grid(True, alpha=0.3)
plt.setp(ax_vol.get_xticklabels(), rotation=45, ha="right")

fig.tight_layout()
st.pyplot(fig)
