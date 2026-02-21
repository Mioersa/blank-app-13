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
st.sidebar.title("📂 Upload Futures Data")
uploaded_files = st.sidebar.file_uploader(
    "Drag & drop multiple intraday CSV files (5‑min futures)",
    type="csv",
    accept_multiple_files=True
)

if not uploaded_files:
    st.warning("👋 Please upload one or more CSV files to begin.")
    st.stop()

# ------------------------------------------------------------
# Load Data
# ------------------------------------------------------------
dfs = []
for uploaded in uploaded_files:
    fn = uploaded.name
    # parse timestamp from file name like ***_ddmmyyyy_hhmmss.csv
    m = re.search(r'_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$', fn)
    ts = datetime.now()
    if m:
        dd, mm, yyyy, HH, MM, SS = m.groups()
        ts = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))

    df = pd.read_csv(uploaded)
    df['timestamp'] = ts + pd.to_timedelta(np.arange(len(df)) * 5, unit='min')
    # ensure contract identification (may already exist inside file)
    if 'contract' not in df.columns:
        df['contract'] = fn
    dfs.append(df)

df = pd.concat(dfs).sort_values(['contract', 'timestamp']).reset_index(drop=True)

# ------------------------------------------------------------
# Filters
# ------------------------------------------------------------
tmin, tmax = df['timestamp'].min(), df['timestamp'].max()
t0, t1 = st.sidebar.slider(
    "Select time window",
    min_value=tmin,
    max_value=tmax,
    value=(tmin, tmax)
)
df = df[(df['timestamp'] >= t0) & (df['timestamp'] <= t1)]

st.title("📊 Intraday Futures Momentum & Correlation Dashboard")
st.caption(f"Data range: {t0} → {t1} | Contracts: {df['contract'].nunique()} | Total rows: {len(df):,}")

# ------------------------------------------------------------
# Core metrics
# ------------------------------------------------------------
df['log_ret'] = np.log(df['closePrice'] / df['closePrice'].shift(1))
df['roc5'] = df['closePrice'].pct_change(5)
df['ma5'] = df['closePrice'].rolling(5).mean()
df['ma20'] = df['closePrice'].rolling(20).mean()
df['macd'] = df['closePrice'].ewm(span=12).mean() - df['closePrice'].ewm(span=26).mean()

# --- Volatility & Intensity ---
df['real_vol'] = df['log_ret'].rolling(10).std()
df['range_pct'] = (df['highPrice'] - df['lowPrice']) / df['closePrice']
df['tr'] = np.maximum.reduce([
    df['highPrice'] - df['lowPrice'],
    (df['highPrice'] - df['closePrice'].shift()).abs(),
    (df['lowPrice'] - df['closePrice'].shift()).abs()
])
df['atr'] = df['tr'].rolling(14).mean()

# --- Derivative clues ---
df['dOI'] = df['openInterest'].diff()
df['dP'] = df['closePrice'].diff()
df['oi_price'] = df['dOI'] * df['dP']
df['spec_ratio'] = df['premiumTurnOver'] / df['totalTurnover']
df['vwap_like'] = df['value'] / df['volume']

# --- Cross relationships ---
df['vol_vol'] = df['volume'].rolling(5).corr(df['real_vol'])

# --- Optional modeling ---
df['ret_lag1'] = df['log_ret'].shift(1)
rolling_window = 50
betas = []
for i in range(len(df)):
    if i < rolling_window:
        betas.append(np.nan)
    else:
        y = df['log_ret'].iloc[i - rolling_window:i]
        x = add_constant(df['ret_lag1'].iloc[i - rolling_window:i])
        model = OLS(y, x, missing='drop').fit()
        betas.append(model.params[-1])
df['mom_strength'] = betas
df['autocorr'] = df['log_ret'].rolling(50).apply(lambda x: x.autocorr(), raw=False)

# ------------------------------------------------------------
# Inter‑contract Correlations
# ------------------------------------------------------------
pivot = df.pivot(index='timestamp', columns='contract', values='log_ret').dropna()
corr_matrix = pivot.corr()
vol_pivot = np.log(pivot.abs() + 1e-9)
vol_correlation = vol_pivot.corr()

# PCA
pca = PCA(n_components=min(2, len(pivot.columns)))
pc = pca.fit_transform(pivot.fillna(0))
pcdf = pd.DataFrame(pc, index=pivot.index, columns=['PC1', 'PC2'])

# Spread analysis (near vs far expiry)
contracts = sorted(df['contract'].unique())
spread_df = pd.DataFrame()
if len(contracts) >= 2:
    near, far = contracts[:2]
    sp = df[df['contract'].isin([near, far])].pivot(index='timestamp', columns='contract', values='closePrice').dropna()
    sp['spread'] = sp[far] - sp[near]
    spread_df = sp

# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------
st.header("Momentum & Direction")
fig_price = px.line(df, x='timestamp', y=['closePrice', 'ma5', 'ma20'], title='Prices & Moving Averages')
st.plotly_chart(fig_price, use_container_width=True)

st.line_chart(df.set_index('timestamp')[['macd', 'roc5']])

st.header("Volatility & Intensity")
st.line_chart(df.set_index('timestamp')[['real_vol', 'atr', 'range_pct']])
st.dataframe(df[['timestamp','contract','real_vol','atr','range_pct']].dropna().tail(10))

st.header("Derivative Clues")
st.line_chart(df.set_index('timestamp')[['oi_price','spec_ratio','vwap_like']])
st.caption("ΔOI·ΔPrice ⇒ >0 long buildup | <0 unwinding.")

st.header("Cross Relationships & Correlations")
st.subheader("Return correlation between contracts")
st.dataframe(corr_matrix.round(3))
st.subheader("Volatility correlation between contracts")
st.dataframe(vol_correlation.round(3))
st.plotly_chart(px.line(pcdf, title="PCA of Returns"), use_container_width=True)

st.subheader("Volume–Volatility rolling correlation (participation intensity)")
st.line_chart(df.set_index('timestamp')['vol_vol'])

if not spread_df.empty:
    st.subheader(f"Spread {contracts[1]} − {contracts[0]}")
    st.line_chart(spread_df['spread'])

st.header("Momentum Persistence & Regression")
st.line_chart(df.set_index('timestamp')[['mom_strength','autocorr']])
st.caption("mom_strength = rolling regression slope (ret_t vs ret_(t−1)), autocorr = persistence")

st.success("✅ All computations complete.")
