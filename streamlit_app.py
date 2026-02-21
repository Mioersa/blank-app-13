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
    st.error("❌ Column 'expiryDate' not found.")
    st.stop()

expiry_options = sorted(first_df["expiryDate"].unique())
selected_expiry = st.selectbox("Select Expiry Date", expiry_options)

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
st.subheader(f"🧾 Combined Data for expiry = {selected_expiry}")
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
sumdf["Δ Volume"] = sumdf["volume"].diff()
sumdf["Δ Price"] = sumdf["last_price"].diff()

# ------------------------------------------------------------
# Indicators
# ------------------------------------------------------------
sumdf["OBV"] = (np.sign(sumdf["Δ Price"].fillna(0)) * sumdf["Δ Volume"].fillna(0)).cumsum()
vol_mean, vol_std = sumdf["Δ Volume"].mean(), sumdf["Δ Volume"].std()
sumdf["spike_flag"] = sumdf["Δ Volume"] > (vol_mean + 2 * vol_std)
sumdf["Cum_Vol"] = sumdf["Δ Volume"].cumsum()
rolling_N = 5
sumdf["RVR"] = sumdf["Δ Volume"] / sumdf["Δ Volume"].rolling(rolling_N, min_periods=1).mean()
short, long = 3, 10
ema_short = sumdf["Δ Volume"].ewm(span=short, adjust=False).mean()
ema_long = sumdf["Δ Volume"].ewm(span=long, adjust=False).mean()
sumdf["Vol_Osc"] = ema_short - ema_long
sumdf["RollCorr"] = sumdf["Δ Price"].rolling(5).corr(sumdf["Δ Volume"])

st.subheader("📊 Volume & Price Indicators Summary")
st.dataframe(sumdf)

# ------------------------------------------------------------
# Chart 1 – with OBV & spikes
# ------------------------------------------------------------
st.subheader("📈 Chart 1 – Last Price (top) & Δ Volume (bottom + OBV + Spikes)")
axis_type = st.radio("Δ Volume Y‑axis scale", ["linear", "log"], horizontal=True, key="yaxis_scale")

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=sumdf["time"], y=sumdf["Δ Volume"],
                     name="Δ Volume", marker_color="orange", opacity=0.6, yaxis="y2"))
fig1.add_trace(go.Scatter(x=sumdf.loc[sumdf["spike_flag"], "time"],
                          y=sumdf.loc[sumdf["spike_flag"], "Δ Volume"],
                          mode="markers", marker=dict(color="red", size=10, symbol="diamond"),
                          name="Spike (>2σ)", yaxis="y2"))
fig1.add_trace(go.Scatter(x=sumdf["time"], y=sumdf["OBV"],
                          mode="lines", line=dict(color="green", width=2, dash="dot"),
                          name="OBV", yaxis="y2"))
fig1.add_trace(go.Scatter(x=sumdf["time"], y=sumdf["last_price"],
                          mode="lines+markers", line=dict(color="blue"),
                          name="Last Price", yaxis="y1"))
fig1.update_layout(
    height=700,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
    yaxis2=dict(domain=[0.0, 0.35], title="Δ Volume / OBV", type=axis_type),
    title=f"Chart 1 – Last Price and Δ Volume with OBV & Spikes — Expiry {selected_expiry}",
    legend=dict(orientation="h"),
    hovermode="x unified"
)
config = {"scrollZoom": True, "displaylogo": False}
st.plotly_chart(fig1, use_container_width=True, config=config)

# ------------------------------------------------------------
# Chart 2 – clean
# ------------------------------------------------------------
st.subheader("📉 Chart 2 – Last Price (top) & Δ Volume (bottom, no OBV / no spikes)")

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=sumdf["time"], y=sumdf["Δ Volume"],
                     name="Δ Volume", marker_color="orange", opacity=0.6, yaxis="y2"))
fig2.add_trace(go.Scatter(x=sumdf["time"], y=sumdf["last_price"],
                          mode="lines+markers", line=dict(color="blue"),
                          name="Last Price", yaxis="y1"))
fig2.update_layout(
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
    yaxis2=dict(domain=[0.0, 0.35], title="Δ Volume", type=axis_type),
    title=f"Chart 2 – Clean Δ Volume Chart",
    legend=dict(orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True, config=config)

# ------------------------------------------------------------
# Classification per interval
# ------------------------------------------------------------
def classify(row):
    slope = np.sign(row["Δ Volume"]) or 0
    corr = np.sign(row["RollCorr"]) if not np.isnan(row["RollCorr"]) else 0
    score = slope * corr
    return 1 if score > 0 else (-1 if score < 0 else 0)

sumdf["Signal_Val"] = sumdf.apply(classify, axis=1)
sumdf["Signal_Label"] = sumdf["Signal_Val"].map({1: "🟢 Bullish", 0: "⚪ Neutral", -1: "🔴 Bearish"})
st.subheader("🧠 Volume Behavior Insights (Per Interval)")
st.dataframe(sumdf[["time", "Δ Volume", "Δ Price", "RollCorr", "Signal_Label"]])

# ------------------------------------------------------------
# Chart 3 – signal line chart
# ------------------------------------------------------------
st.subheader("🩺 Chart 3 – Signal (+1/0/−1)")
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=sumdf["time"], y=sumdf["Signal_Val"],
    mode="lines+markers", line=dict(color="purple", width=3),
    name="Signal"
))
fig3.update_layout(
    height=300,
    yaxis=dict(title="Signal", tickvals=[-1, 0, 1],
               ticktext=["Bear (-1)", "Neutral (0)", "Bull (+1)"]),
    xaxis=dict(title="Capture Time (HH:MM)"),
    title="Chart 3 – Per‑Interval Signal Line",
    hovermode="x unified"
)
st.plotly_chart(fig3, use_container_width=True, config=config)

# ------------------------------------------------------------
# Chart 4 – Clean chart with colored signal dots
# ------------------------------------------------------------
st.subheader("🎯 Chart 4 – Δ Volume Bars + Per‑Interval Signal Overlay")

fig4 = go.Figure()
# Bars
fig4.add_trace(go.Bar(
    x=sumdf["time"], y=sumdf["Δ Volume"],
    name="Δ Volume", marker_color="orange", opacity=0.5, yaxis="y2"
))
# Overlay dots by signal
colors = sumdf["Signal_Val"].map({1: "green", 0: "gray", -1: "red"})
fig4.add_trace(go.Scatter(
    x=sumdf["time"], y=sumdf["Δ Volume"],
    mode="markers", marker=dict(color=colors, size=10, line=dict(width=1, color="black")),
    name="Signal (Bull/Neutral/Bear)", yaxis="y2"
))
# Price on top
fig4.add_trace(go.Scatter(
    x=sumdf["time"], y=sumdf["last_price"],
    mode="lines+markers", line=dict(color="blue"),
    name="Last Price", yaxis="y1"
))
fig4.update_layout(
    height=600,
    margin=dict(l=60, r=40, t=60, b=60),
    xaxis=dict(title="Capture Time (HH:MM)", rangeslider=dict(visible=True)),
    yaxis=dict(domain=[0.45, 1.0], title="Last Price"),
    yaxis2=dict(domain=[0.0, 0.35], title="Δ Volume", type=axis_type),
    title="Chart 4 – Last Price & Δ Volume with Signal Dots",
    legend=dict(orientation="h"),
    hovermode="x unified"
)
st.plotly_chart(fig4, use_container_width=True, config=config)
# ------------------------------------------------------------
# Chart 5 – RVR and Volume Oscillator signals
# ------------------------------------------------------------
st.subheader("⚙️ Chart 5 – RVR (Relative Vol Ratio) and Volume Oscillator Signals")

fig5 = go.Figure()

# RVR bars (color by strength)
fig5.add_trace(go.Bar(
    x=sumdf["time"], y=sumdf["RVR"],
    marker

# ------------------------------------------------------------
# Overall signal
# ------------------------------------------------------------
last_rows = sumdf.tail(5)
slope = np.sign(last_rows["Cum_Vol"].iloc[-1] - last_rows["Cum_Vol"].iloc[0])
corr_sign = np.sign(sumdf["RollCorr"].iloc[-1])
score = slope * corr_sign
if score > 0:
    overall_signal = "🟢 **Bullish Accumulation**"
elif score < 0:
    overall_signal = "🔴 **Bearish Distribution**"
else:
    overall_signal = "⚪ **Neutral / Indecisive**"

st.subheader("📈 Overall Auto‑Signal")
st.markdown(overall_signal)
st.success("✅ All 4 Charts + Signal Overlay Complete.")


