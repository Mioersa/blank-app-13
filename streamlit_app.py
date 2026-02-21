import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from sklearn.decomposition import PCA
from statsmodels.api import OLS, add_constant
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
    st.warning("👋 Upload at least one CSV to start preview.")
    st.stop()

# ------------------------------------------------------------
# Load & Preview
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

raw_df = pd.concat(dfs).sort_values(["contract", "timestamp"]).reset_index(drop=True)

st.subheader("📄 Data Preview")
st.dataframe(raw_df.head(20))
st.caption(f"{len(raw_df):,} rows | {raw_df['contract'].nunique()} contracts | {raw_df['expiryDate'].nunique()} expiries")

# ------------------------------------------------------------
# Run analytics trigger
# ------------------------------------------------------------
if not st.button("➡️ Run analytics"):
    st.stop()

df = raw_df.copy()
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ------------------------------------------------------------
# Core indicators
# ------------------------------------------------------------
df["log_ret"] = np.log(df["closePrice"] / df["closePrice"].shift(1))
df["roc5"] = df["closePrice"].pct_change(5)
df["ma5"] = df["closePrice"].rolling(5).mean()
df["ma20"] = df["closePrice"].rolling(20).mean()
df["macd"] = df["closePrice"].ewm(span=12).mean() - df["closePrice"].ewm(span=26).mean()

df["real_vol"] = df["log_ret"].rolling(10).std()
df["range_pct"] = (df["highPrice"] - df["lowPrice"]) / df["closePrice"]
df["tr"] = np.maximum.reduce([
    df["highPrice"] - df["lowPrice"],
    (df["highPrice"] - df["closePrice"].shift()).abs(),
    (df["lowPrice"] - df["closePrice"].shift()).abs(),
])
df["atr"] = df["tr"].rolling(14).mean()

df["dOI"] = df["openInterest"].diff()
df["dP"] = df["closePrice"].diff()
df["oi_price"] = df["dOI"] * df["dP"]
df["spec_ratio"] = df["premiumTurnOver"] / df["totalTurnover"]
df["vwap_like"] = df["value"] / df["volume"]
df["vol_vol"] = df["volume"].rolling(5).corr(df["real_vol"])

# Rolling regression
df["ret_lag1"] = df["log_ret"].shift(1)
betas = []
window = 50
for i in range(len(df)):
    if i < window:
        betas.append(np.nan)
        continue
    y = df["log_ret"].iloc[i - window:i].dropna()
    x = df["ret_lag1"].iloc[i - window:i].dropna()
    common = x.index.intersection(y.index)
    if len(common) < 5:
        betas.append(np.nan)
        continue
    y, x = y.loc[common], add_constant(x.loc[common])
    try:
        model = OLS(y, x).fit()
        betas.append(model.params[-1])
    except Exception:
        betas.append(np.nan)
df["mom_strength"] = betas
df["autocorr"] = df["log_ret"].rolling(50).apply(lambda x: x.autocorr(), raw=False)

# ------------------------------------------------------------
# Correlations & PCA
# ------------------------------------------------------------
pivot = df.pivot(index="timestamp", columns="contract", values="log_ret").dropna()
corr_matrix, vol_correlation, pcdf = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(index=pivot.index)
if not pivot.empty:
    corr_matrix = pivot.corr()
    vol_pivot = np.log(np.abs(pivot) + 1e-9)
    vol_correlation = vol_pivot.corr()
    if pivot.shape[0] > 1 and pivot.shape[1] > 1:
        try:
            pca = PCA(n_components=2)
            comp = pca.fit_transform(pivot.fillna(0))
            pcdf = pd.DataFrame(comp, index=pivot.index, columns=["PC1", "PC2"])
        except Exception as e:
            st.warning(f"PCA skipped: {e}")

# ------------------------------------------------------------
# Spread (near/far expiry)
# ------------------------------------------------------------
contracts = sorted(df["contract"].unique())
spread_df = pd.DataFrame()
if len(contracts) >= 2:
    near, far = contracts[:2]
    spread_df = (
        df[df["contract"].isin([near, far])]
        .pivot(index="timestamp", columns="contract", values="closePrice")
        .dropna()
    )
    spread_df["spread"] = spread_df[far] - spread_df[near]

# ------------------------------------------------------------
# Streamlit Charts
# ------------------------------------------------------------
st.header("Momentum & Direction")
st.line_chart(df.set_index("timestamp")[["closePrice", "ma5", "ma20"]])
st.line_chart(df.set_index("timestamp")[["macd", "roc5"]])

st.header("Volatility & Intensity")
st.line_chart(df.set_index("timestamp")[["real_vol", "atr", "range_pct"]])

st.header("Derivative Clues")
st.line_chart(df.set_index("timestamp")[["oi_price", "spec_ratio", "vwap_like"]])

if not pivot.empty:
    st.header("Cross‑Contract Correlations")
    st.dataframe(corr_matrix.round(3))
    st.dataframe(vol_correlation.round(3))

if not pcdf.empty:
    st.line_chart(pcdf)

# ------------------------------------------------------------
# Last Price vs Time (markers)
# ------------------------------------------------------------
contracts = sorted(df["contract"].unique())
selected = contracts[0] if len(contracts) else None

if selected:
    sub = df[df["contract"] == selected][["timestamp", "lastPrice"]].dropna().copy()
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sub["timestamp"], sub["lastPrice"], color="tab:orange", linewidth=1.2)
        ax.set_title(f"{selected} — Last Price vs Captured Times")
        ax.set_ylabel("Last Price")
        ax.grid(True, alpha=0.3)
        for t, lbl in zip(upload_times, upload_labels):
            ax.axvline(t, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(t, sub["lastPrice"].min(), lbl, rotation=90,
                    va="bottom", ha="center", fontsize=8, color="gray")
        ax.set_xticks([])
        st.pyplot(fig)

# ------------------------------------------------------------
# Volume Change Between Files (chart first, table below)
# ------------------------------------------------------------
st.header("Volume Change Between Files")

records = []
prev_vol_last = np.nan

for i, df_part in enumerate(dfs):
    curr_first = df_part["volume"].iloc[0] if "volume" in df_part else np.nan
    if np.isnan(prev_vol_last):
        delta = 0
    else:
        delta = curr_first - prev_vol_last

    records.append({
        "capture_time": upload_times[i],
        "label": upload_labels[i],
        "previous_volume_last": prev_vol_last,
        "current_volume_first": curr_first,
        "delta_volume": delta
    })
    prev_vol_last = curr_first  # chain update

vol_change_table = pd.DataFrame(records)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 3))
ax.bar(vol_change_table["capture_time"], vol_change_table["delta_volume"],
       width=0.001, color="tab:blue", alpha=0.75)
for t, lbl in zip(upload_times, upload_labels):
    ax.text(t, 0, lbl, rotation=90, va="bottom", ha="center",
            fontsize=8, color="red")
ax.set_title("Δ Volume Between Captures (Chained Comparison)")
ax.set_ylabel("Change In Volume")
ax.grid(True, alpha=0.3)
ax.set_xticks([])
st.pyplot(fig)

# --- Table (below chart) ---
st.subheader("📊 Detailed Volume Change Data")
st.dataframe(vol_change_table[["capture_time", "label",
                               "previous_volume_last",
                               "current_volume_first",
                               "delta_volume"]])

# ------------------------------------------------------------
# Final
# ------------------------------------------------------------
st.header("Momentum Persistence & Regression")
st.line_chart(df.set_index("timestamp")[["mom_strength", "autocorr"]])
st.success("✅ All computations finished (Streamlit + Matplotlib only; no WebGL).")
