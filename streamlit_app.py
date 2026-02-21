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
        label = f"{HH}:{MM}"
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
selected_expiry = st.selectbox("Select Expiry Date", expiry_options)

# ------------------------------------------------------------
# Filter all files for selected expiryDate
# ------------------------------------------------------------
filtered = []
for i, df_file in enumerate(dfs):
    sub = df_file[df_file["expiryDate"] == selected_expiry].copy()
    if sub.empty:
        continue
    sub["label"] = upload_labels[i]
    sub["capture_time"] = upload_times[i]
    filtered.append(sub)

if not filtered:
    st.warning("No rows found for that expiry.")
    st.stop()

final_df = pd.concat(filtered).sort_values(["contract", "timestamp"]).reset_index(drop=True)
st.subheader(f"🧾 Combined Data for expiry = {selected_expiry}")
st.dataframe(final_df)

# ------------------------------------------------------------
# Build summary (Δ Volume per capture)
# ------------------------------------------------------------
records = []
for lbl in upload_labels:
    sub = final_df[final_df["label"] == lbl]
    if sub.empty:
        continue
    vol = sub["volume"].iloc[0] if "volume" in sub else np.nan
    price = sub["lastPrice"].iloc[-1] if "lastPrice" in sub else np.nan
    records.append({"time": lbl, "volume": vol, "last_price": price})

sumdf = pd.DataFrame(records)
sumdf["Δ Volume"] = sumdf["volume"].diff()
st.subheader("📊 Volume & Last Price Summary")
st.dataframe(sumdf)

# ------------------------------------------------------------
# Plotly: scrollable horizontally + zoom/pan vertically
# ------------------------------------------------------------
st.subheader("📈 Last Price (top) & Δ Volume (bottom 30%) Chart (Zoomable Y)")

fig = go.Figure()

# bottom Δ Volume bars
fig.add_trace(go.Bar(
    x=sumdf["time"],
    y=sumdf["Δ Volume"],
    name="Δ Volume",
    marker_color="orange",
    opacity=0.6,
    yaxis="y2",
))

# top Last Price line
fig.add_trace(go.Scatter(
    x=sumdf["time"],
    y=sumdf["last_price"],
    mode="lines+markers",
    name="Last Price",
    line=dict(color="blue"),
    yaxis="y1",
))

fig.update_layout(
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(
        title="Capture Time (HH:MM)",
        rangeslider=dict(visible=True),  # horizontal scroll
    ),
    yaxis=dict(
        domain=[0.35, 1.0],
        title="Last Price",
        tickformat=".0f",
        separatethousands=True,
        rangemode="normal",
    ),
    yaxis2=dict(
        domain=[0.0, 0.3],
        title="Δ Volume",
        rangemode="tozero",
    ),
    title=f"Last Price and Δ Volume — Expiry {selected_expiry}",
    legend=dict(orientation="h"),
    hovermode="x unified",
)

# enable free y‑zoom/pan
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

st.success("✅ Scrollable & Zoomable chart ready.")
