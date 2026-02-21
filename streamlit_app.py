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
# Show combined data table
# ------------------------------------------------------------
st.subheader(f"🧾 Combined Data for expiryDate = {selected_expiry}")
st.dataframe(final_df)

# ------------------------------------------------------------
# Build summary table: time(label), volume, lastPrice, Δ Volume
# ------------------------------------------------------------
summary_records = []
for lbl in upload_labels:
    sub = final_df[final_df["label"] == lbl]
    if sub.empty:
        continue
    vol_val = sub["volume"].iloc[0] if "volume" in sub else np.nan
    price_val = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
    summary_records.append({
        "time": lbl,
        "volume": vol_val,
        "last_price": price_val,
    })

summary_df = pd.DataFrame(summary_records)
summary_df["Δ Volume"] = summary_df["volume"].diff()

st.subheader("📊 Volume & Last Price Summary")
st.dataframe(summary_df)

# ------------------------------------------------------------
# Plot: restrict Δ Volume bars to ~lower 30% of Y‑axis
# ------------------------------------------------------------
st.subheader("📈 Last Price vs Δ Volume Chart (Δ Volume scaled to 30%)")

# determine scaling factor relative to last price range
if not summary_df["last_price"].isna().all():
    price_range = summary_df["last_price"].max() - summary_df["last_price"].min()
else:
    price_range = 1
vol_range = summary_df["Δ Volume"].abs().max() if not summary_df["Δ Volume"].isna().all() else 1
# scale ΔVol to 30% of price range
scale_factor = 0.3 * price_range / vol_range if vol_range != 0 else 1

fig, ax1 = plt.subplots(figsize=(10, 4))

# scaled bar heights centered near min price
min_price = summary_df["last_price"].min() if not summary_df["last_price"].isna().all() else 0
scaled_vol = summary_df["Δ Volume"] * scale_factor + (min_price - 0.3 * price_range * 0.2)

ax1.bar(summary_df["time"], scaled_vol,
        color="orange", alpha=0.5, label="Δ Volume (scaled)")
ax1.set_ylim(min_price - 0.35 * price_range, summary_df["last_price"].max() * 1.05)

# overlay last price line on top
ax1.plot(summary_df["time"], summary_df["last_price"],
         color="tab:blue", marker="o", label="Last Price")

ax1.set_xlabel("Capture Label (Time)")
ax1.set_ylabel("Value / Δ Volume (scaled)")
ax1.set_title(f"Last Price vs Δ Volume — Expiry {selected_expiry}")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

fig.tight_layout()
st.pyplot(fig)
