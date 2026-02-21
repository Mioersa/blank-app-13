import streamlit as st
import pandas as pd
import numpy as np
import glob, os, re
import plotly.express as px
from datetime import datetime
from sklearn.decomposition import PCA
from statsmodels.api import OLS, add_constant

st.set_page_config(page_title="Intraday Futures Analytics", layout="wide")

@st.cache_data
def load_data(pattern):
    files = glob.glob(pattern)
    dfs = []
    for fp in files:
        fn = os.path.basename(fp)
        m = re.search(r'_(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})\.csv$', fn)
        ts = datetime.now()
        if m:
            dd, mm, yyyy, HH, MM, SS = m.groups()
            ts = datetime(int(yyyy), int(mm), int(dd), int(HH), int(MM), int(SS))
        df = pd.read_csv(fp)
        df['timestamp'] = ts + pd.to_timedelta(np.arange(len(df))*5, unit='min')
        df['contract'] = df.get('contract', os.path.splitext(fn)[0])
        dfs.append(df)
    df = pd.concat(dfs).sort_values(['contract','timestamp']).reset_index(drop=True)
    return df

# --- Load ---
df = load_data("*.csv")
st.sidebar.header("Filters")
tmin, tmax = st.sidebar.slider("Time range",
    min_value=df['timestamp'].min(),
    max_value=df['timestamp'].max(),
    value=(df['timestamp'].min(), df['timestamp'].max()))
df = df[(df['timestamp']>=tmin)&(df['timestamp']<=tmax)]

st.title("📊 Intraday Futures Momentum & Correlation Analysis")

# --- Base measures ---
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
df['oi_price'] = df['dOI']*df['dP']
df['spec_ratio'] = df['premiumTurnOver'] / df['totalTurnover']
df['vwap_like'] = df['value'] / df['volume']

# --- Intensity relationships ---
df['vol_vol'] = df['volume'].rolling(5).corr(df['real_vol'])

# --- Optional modeling ---
# Rolling regression ret_t vs ret_(t-1)
df['ret_lag1'] = df['log_ret'].shift(1)
window = 50
betas = []
for i in range(len(df)):
    if i < window:
        betas.append(np.nan)
    else:
        y = df['log_ret'].iloc[i-window:i]
        x = add_constant(df['ret_lag1'].iloc[i-window:i])
        model = OLS(y, x, missing='drop').fit()
        betas.append(model.params[-1])
df['mom_strength'] = betas

# Autocorr rolling
df['autocorr'] = df['log_ret'].rolling(50).apply(lambda x: x.autocorr(), raw=False)

# --- Cross correlations ---
pivot = df.pivot(index='timestamp', columns='contract', values='log_ret').dropna()
corr_matrix = pivot.corr()

# Vol clustering/spillover correlation of log(abs(ret))
vol_pivot = np.log(pivot.abs()+1e-9)
vol_correlation = vol_pivot.corr()

# PCA
pca = PCA(n_components=min(2,len(pivot.columns)))
pc = pca.fit_transform(pivot.fillna(0))
pcdf = pd.DataFrame(pc, index=pivot.index, columns=['PC1','PC2'])

# --- Spread (near vs far) ---
contracts = sorted(df['contract'].unique())
if len(contracts)>=2:
    near, far = contracts[:2]
    sp = df[df['contract'].isin([near, far])].pivot(index='timestamp', columns='contract', values='closePrice').dropna()
    sp['spread'] = sp[far] - sp[near]
else:
    sp = pd.DataFrame()

# --- UI sections ---
st.header("Momentum & Price Direction")
fig = px.line(df, x='timestamp', y=['closePrice','ma5','ma20'], title='Close & Moving Averages')
st.plotly_chart(fig, use_container_width=True)
st.line_chart(df.set_index('timestamp')[['macd','roc5']])

st.header("Volatility & Intensity")
st.line_chart(df.set_index('timestamp')[['real_vol','atr','range_pct']])
st.dataframe(df[['timestamp','contract','real_vol','atr','range_pct']].dropna().tail(10))

st.header("Derivative Clues")
st.line_chart(df.set_index('timestamp')[['oi_price','spec_ratio','vwap_like']])
st.write("Interpretation: ΔOI*ΔP > 0 ⇒ Long buildup; < 0 ⇒ Unwinding")

st.header("Correlation Across Contracts")
st.subheader("Return Correlation Matrix")
st.dataframe(corr_matrix.round(3))
st.subheader("Volatility (log|ret|) Correlation Matrix")
st.dataframe(vol_correlation.round(3))

st.subheader("PCA Components")
st.plotly_chart(px.line(pcdf, title="PCA of Returns"), use_container_width=True)

st.header("Cross Relationships")
st.write("Volume–Volatility Rolling Correlation")
st.line_chart(df.set_index('timestamp')['vol_vol'])

if not sp.empty:
    st.subheader(f"Spread: {contracts[1]} − {contracts[0]}")
    st.line_chart(sp['spread'])

st.header("Optional Modeling Insights")
st.line_chart(df.set_index('timestamp')[['mom_strength','autocorr']])
st.write("`mom_strength`: rolling regression slope ret_t vs ret_(t‑1)")

st.success("✅ All computations implemented: Momentum, Volatility, Intensity, Cross Corr, Derivative Clues, Rolling Regression, Autocorr, PCA, Spread.")


