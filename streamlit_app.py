import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(page_title="Intraday Futures Analytics", layout="wide")

# ------------------------------------------------------------
# Upload
# ------------------------------------------------------------
st.sidebar.title("📂 Upload Futures CSVs")
uploaded_files = st.sidebar.file_uploader(
    "Drag‑drop intraday futures CSVs (5‑min interval)",
    type="csv", accept_multiple_files=True,
)
if not uploaded_files:
    st.warning("👋 Upload at least one CSV.")
    st.stop()

# ------------------------------------------------------------
# Load files
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
# Expiry dropdown
# ------------------------------------------------------------
first_df = dfs[0]
if "expiryDate" not in first_df.columns:
    st.error("❌ Column 'expiryDate' not found.")
    st.stop()

expiry_options = sorted(first_df["expiryDate"].unique())
selected_expiry = st.selectbox("Select Expiry Date", expiry_options)

# ------------------------------------------------------------
# Filter for expiry
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
    st.warning("No data for chosen expiry.")
    st.stop()

final_df = pd.concat(filtered).sort_values(["contract", "timestamp"]).reset_index(drop=True)

st.subheader(f"🧾 Combined Data for expiry = {selected_expiry}")
st.dataframe(final_df)

# ------------------------------------------------------------
# Core summary
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
sumdf["Δ Price"] = sumdf["last_price"].diff()

# ------------------------------------------------------------
# Extra indicators
# ------------------------------------------------------------
# 1️⃣ OBV (On-Balance Volume)
sumdf["OBV"] = (np.sign(sumdf["Δ Price"].fillna(0)) * sumdf["Δ Volume"].fillna(0)).cumsum()

# 2️⃣ Spikes (Threshold > mean + 2*STD)
vol_mean, vol_std = sumdf["Δ Volume"].mean(), sumdf["Δ Volume"].std()
sumdf["spike_flag"] = sumdf["Δ Volume"] > (vol_mean + 2 * vol_std)

# 3️⃣ Cumulative Volume
sumdf["Cum_Vol"] = sumdf["Δ Volume"].cumsum()

# 4️⃣ Relative Volume Ratio (RVR)
rolling_N = 5
sumdf["RVR"] = sumdf["Δ Volume"] / sumdf["Δ Volume"].rolling(rolling_N, min_periods=1).mean()

# 5️⃣ Volume Oscillator (short vs. long EMA)
short, long = 3, 10
ema_short = sumdf["Δ Volume"].ewm(span=short, adjust=False).mean()
ema_long = sumdf["Δ Volume"].ewm(span=long, adjust=False).mean()
sumdf["Vol_Osc"] = ema_short - ema_long

# 6️⃣ Rolling correlation Δ Price vs Δ Volume
sumdf["RollCorr"] = sumdf["Δ Price"].rolling(5).corr(sumdf["Δ Volume"])

st.subheader("📊 Volume & Price Indicators Summary")
st.dataframe(sumdf)

# ------------------------------------------------------------
# Plot (Plotly)
# ------------------------------------------------------------
st.subheader("📈 Last Price (top) & Δ Volume (bottom + OBV overlay + Spike markers)")

fig = go.Figure()

# bottom Δ Volume bars
fig.add_trace(go.Bar(
    x=sumdf["time"], y=sumdf["Δ Volume"],
    name="Δ Volume", marker_color="orange",
    opacity=0.6, yaxis="y2",
))

# Spike markers
fig.add_trace(go.Scatter(
    x=sumdf.loc[sumdf["spike_flag"], "time"],
    y=sumdf.loc[sumdf["spike_flag"], "Δ Volume"],
    mode="markers", marker=dict(color="red", size=10, symbol="diamond"),
    name="Spike (> 2σ)", yaxis="y2",
))

# OBV line under volume axis
fig.add_trace(go.Scatter(
    x=sumdf["time"], y=sumdf["OBV"],
    mode="lines", line=dict(color="green", width=2, dash="dot"),
    name="OBV", yaxis="y2",
))

# top Last Price line
fig.add_trace(go.Scatter(
    x=sumdf["time"], y=sumdf["last_price"],
    mode="lines+markers", name="Last Price",
    line=dict(color="blue"), yaxis="y1",
))

fig.update_layout(
    height=650,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(
        title="Capture Time (HH:MM)",
        rangeslider=dict(visible=True),
    ),
    yaxis=dict(
        domain=[0.45, 1.0], title="Last Price",
        tickformat=".0f", separatethousands=True,
    ),
    yaxis2=dict(
        domain=[0.0, 0.35], title="Δ Volume / OBV", rangemode="tozero",
    ),
    title=f"Last Price and Δ Volume with OBV & Spikes — Expiry {selected_expiry}",
    legend=dict(orientation="h"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

# ------------------------------------------------------------
# Human‑readable interpretation
# ------------------------------------------------------------
st.subheader("🧠 Volume Behavior Insights")

desc = []
# Accumulation trend
if sumdf["Cum_Vol"].iloc[-1] > 0:
    desc.append("✅ Net positive cumulative volume → overall **accumulation**.")
else:
    desc.append("⚠️ Net negative cumulative volume → possible **distribution**.")

# Spike activity
spike_ratio = sumdf["spike_flag"].mean()
if spike_ratio > 0.2:
    desc.append("🚨 Frequent volume spikes → **high speculative bursts**.")
elif spike_ratio > 0.05:
    desc.append("⏫ Occasional spikes → moderate bursts of activity.")
else:
    desc.append("🟢 Few spikes → stable flow.")

# RVR
rvr_mean = sumdf["RVR"].mean()
if rvr_mean > 1.5:
    desc.append("🔥 Average RVR > 1.5 → volumes mostly above normal (high participation).")
elif rvr_mean < 0.8:
    desc.append("💤 Low RVR → below‑average engagement.")
else:
    desc.append("⚖️ Normal RVR → balanced trade flow.")

# Volume Oscillator
osc_last = sumdf["Vol_Osc"].iloc[-1]
if osc_last > 0:
    desc.append("📈 Positive volume oscillator → **momentum pickup**.")
else:
    desc.append("📉 Negative oscillator → **waning momentum**.")

# Rolling correlation
corr_last = sumdf["RollCorr"].iloc[-1]
if corr_last > 0.5:
    desc.append("📊 Strong + corr(Δ Price, Δ Volume) → volume confirming trend.")
elif corr_last < -0.3:
    desc.append("⚡ Negative correlation → possible exhaustion / divergence.")
else:
    desc.append("😐 Neutral corr → sideways or mixed participation.")

st.markdown("\n".join(desc))

st.success("✅ OBV + spike overlay + readable volume signals complete.")
