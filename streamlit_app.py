import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go

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
        label = f"{HH}:{MM}"              # <-- keep only HH:MM
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
    summary_records.append({
        "time": lbl,
        "volume": vol_val,
        "last_price": price_val,
    })

summary_df = pd.DataFrame(summary_records)
summary_df["Δ Volume"] = summary_df["volume"].diff()

# ------------------------------------------------------------
# Display summary table
# ------------------------------------------------------------
st.subheader("📊 Volume & Last Price Summary")
st.dataframe(summary_df)

# ------------------------------------------------------------
# Plotly scrollable chart
# ------------------------------------------------------------
st.subheader("📈 Scrollable Last Price (top) & Δ Volume (bottom 30%) Chart")

fig = go.Figure()

# Δ Volume (bar, bottom pane)
fig.add_trace(
    go.Bar(
        x=summary_df["time"],
        y=summary_df["Δ Volume"],
        name="Δ Volume",
        marker_color="orange",
        opacity=0.6,
        yaxis="y2",
    )
)

# Last Price (line, top)
fig.add_trace(
    go.Scatter(
        x=summary_df["time"],
        y=summary_df["last_price"],
        mode="lines+markers",
        name="Last Price",
        line=dict(color="blue"),
        yaxis="y1",
    )
)

# --- Setup two y-axes: y1 top 70%, y2 bottom 30% ---
fig.update_layout(
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),  # <-- Scroll here
    yaxis=dict(domain=[0.35, 1.0], title="Last Price", showgrid=True),
    yaxis2=dict(domain=[0.0, 0.3], title="Δ Volume", showgrid=True),
    title=f"Last Price and Δ Volume — Expiry {selected_expiry}",
    legend=dict(orientation="h"),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)
