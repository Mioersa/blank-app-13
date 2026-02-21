import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
import plotly.express as px
from sklearn.decomposition import PCA
from statsmodels.api import OLS, add_constant

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
# Stage 1 – Preview raw data
# ------------------------------------------------------------
dfs = []
for uploaded in uploaded_files:
    fn = uploaded.name
    # extract timestamp from filename ***_ddmmyyyy_hhmmss.csv
    m = re.search(r"_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$", fn)
    base_time = datetime.now()
    if m:
        dd, mm, yyyy, HH, MM, SS = m.groups()
        base_time = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))

    df = pd.read_csv(uploaded)
    df["timestamp"] = base_time + pd.to_timedelta(np.arange(len(df)) * 5, unit="min")
    dfs.append(df)

raw_df = pd.concat(dfs).sort_values(["contract", "timestamp"]).reset_index(drop=True)

st.subheader("📄 Data Preview")
st.dataframe(raw_df.head(20))
st.caption(f"Rows: {len(raw_df):,} | Contracts: {raw_df['contract'].nunique()} | Expiries: {raw_df['expiryDate'].nunique()}")

# ------------------------------------------------------------
# Run analytics only when user clicks
# ------------------------------------------------------------
if not st.button("➡️ Run analytics"):
    st.stop()

df = raw_df.copy()

# ------------------------------------------------------------
# Safe time‑range filter
# ------------------------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"])
if df["timestamp"].nunique() > 1:
    tmin, tmax = df["timestamp"].min().to_pydatetime(), df["timestamp"].max().to_pydatetime()
    t0, t1 = st.sidebar.slider(
        "Select time window",
        min_value=tmin,
        max_value=tmax,
        value=(tmin, tmax),
        format="YYYY‑MM‑DD HH:mm",
    )
    df = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]

st.title("📊 Intraday Futures Momentum & Correlation Dashboard")

# ------------------------------------------------------------
# Core indicators
# ------------------------------------------------------------
df["log_ret"] = np.log(df["closePrice"] / df["closePrice"].shift(1))
df["roc5"] = df["closePrice"].pct_change(5)
df["ma5"] = df["closePrice"].rolling(5).mean()
df["ma20"] = df["closePrice"].rolling(20).mean()
df["macd"] = df["closePrice"].ewm(span=12).mean() - df["closePrice"].ewm(span=26).mean()

# Volatility & Intensity
df["real_vol"] = df["log_ret"].rolling(10).std()
df["range_pct"] = (df["highPrice"] - df["lowPrice"]) / df["closePrice"]
df["tr"] = np.maximum.reduce([
    df["highPrice"] - df["lowPrice"],
    (df["highPrice"] - df["closePrice"].shift()).abs(),
    (df["lowPrice"] - df["closePrice"].shift()).abs(),
])
df["atr"] = df["tr"].rolling(14).mean()

# Derivative clues
df["dOI"] = df["openInterest"].diff()
df["dP"] = df["closePrice"].diff()
df["oi_price"] = df["dOI"] * df["dP"]
df["spec_ratio"] = df["premiumTurnOver"] / df["totalTurnover"]
df["vwap_like"] = df["value"] / df["volume"]

# Cross relationships
df["vol_vol"] = df["volume"].rolling(5).corr(df["real_vol"])

# ------------------------------------------------------------
# Rolling regression (robust)
# ------------------------------------------------------------
df["ret_lag1"] = df["log_ret"].shift(1)
betas = []
window = 50
for i in range(len(df)):
    if i < window:
        betas.append(np.nan)
        continue
    y = df["log_ret"].iloc[i - window : i].dropna()
    x = df["ret_lag1"].iloc[i - window : i].dropna()
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
# Cross‑contract correlations & PCA
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

# Spread near/far expiry
contracts = sorted(df["contract"].unique())
spread_df = pd.DataFrame()
if len(contracts) >= 2:
    near, far = contracts[:2]
    sp = (
        df[df["contract"].isin([near, far])]
        .pivot(index="timestamp", columns="contract", values="closePrice")
        .dropna()
    )
    sp["spread"] = sp[far] - sp[near]
    spread_df = sp

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
st.header("Momentum & Direction")
st.plotly_chart(
    px.line(df, x="timestamp", y=["closePrice", "ma5", "ma20"], title="Close & Moving Averages"),
    use_container_width=True,
)
st.line_chart(df.set_index("timestamp")[["macd", "roc5"]])

st.header("Volatility & Intensity")
st.line_chart(df.set_index("timestamp")[["real_vol", "atr", "range_pct"]])

st.header("Derivative Clues")
st.line_chart(df.set_index("timestamp")[["oi_price", "spec_ratio", "vwap_like"]])

if not pivot.empty:
    st.header("Cross‑Contract Correlations")
    st.subheader("Return Correlation Matrix")
    st.dataframe(corr_matrix.round(3))
    st.subheader("Volatility Correlation Matrix")
    st.dataframe(vol_correlation.round(3))
    if not pcdf.empty:
        st.plotly_chart(px.line(pcdf, title="PCA of Returns"), use_container_width=True)

if not spread_df.empty:
    st.subheader(f"Spread {contracts[1]} − {contracts[0]}")
    st.line_chart(spread_df["spread"])

st.header("Momentum Persistence & Regression")
st.line_chart(df.set_index("timestamp")[["mom_strength", "autocorr"]])

st.success("✅ All computations finished — charts rendered successfully.")
