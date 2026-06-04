import joblib
import pandas as pd
import numpy as np
import os

# ─────────────────────────────────────────────
# 1. LOAD MODEL & SCALER (separate files)
# ─────────────────────────────────────────────
model = joblib.load("gold_5m_model.pkl")
scaler = joblib.load("gold_5m_scaler.pkl")

# ─────────────────────────────────────────────
# 2. LOAD DATA
# ─────────────────────────────────────────────
csv_path = "data/raw/xauusd_5m.csv"
if not os.path.exists(csv_path):
    print(f"❌ CSV not found: {csv_path}")
    exit(1)

df = pd.read_csv(csv_path)
df.columns = [c.strip().lower() for c in df.columns]

# Convert datetime
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

# Convert boolean columns
bool_cols = ['bullish_fvg','bearish_fvg','buy_side_sweep','sell_side_sweep','bullish_mss','bearish_mss']
for col in bool_cols:
    if col in df.columns:
        df[col] = df[col].map(lambda x: 1 if str(x).strip().lower() in ['true','1','yes'] else 0)

# ─────────────────────────────────────────────
# 3. ADD FEATURES (same as in training)
# ─────────────────────────────────────────────
c, h, l, o = df['close'], df['high'], df['low'], df['open']

# Momentum
df['return_1']  = c.pct_change(1)
df['return_3']  = c.pct_change(3)
df['return_6']  = c.pct_change(6)
df['return_12'] = c.pct_change(12)
df['return_24'] = c.pct_change(24)

# Candle shape
df['body']       = (c - o) / (c + 1e-10)
df['upper_wick'] = (h - c.clip(lower=o)) / (h - l + 1e-10)
df['lower_wick'] = (c.clip(upper=o) - l) / (h - l + 1e-10)
df['hl_range']   = (h - l) / c

# RSI
delta = c.diff()
for n, name in [(7,'rsi_7'), (14,'rsi_14')]:
    g  = delta.clip(lower=0).rolling(n).mean()
    ls = (-delta.clip(upper=0)).rolling(n).mean()
    df[name] = 100 - (100 / (1 + g / (ls + 1e-10)))

# MACD
ema12 = c.ewm(span=12).mean()
ema26 = c.ewm(span=26).mean()
df['macd']       = ema12 - ema26
df['macd_sig']   = df['macd'].ewm(span=9).mean()
df['macd_hist']  = df['macd'] - df['macd_sig']
df['macd_cross'] = np.sign(df['macd_hist']) - np.sign(df['macd_hist'].shift(1))

# Bollinger Bands
sma20 = c.rolling(20).mean()
std20 = c.rolling(20).std()
df['bb_pos']   = (c - (sma20 - 2*std20)) / (4*std20 + 1e-10)
df['bb_width'] = 4 * std20 / sma20

# Stochastic
low9  = l.rolling(9).min()
high9 = h.rolling(9).max()
df['stoch_k']    = 100 * (c - low9) / (high9 - low9 + 1e-10)
df['stoch_d']    = df['stoch_k'].rolling(3).mean()
df['stoch_cross']= np.sign(df['stoch_k'] - df['stoch_d']) - np.sign(df['stoch_k'].shift(1) - df['stoch_d'].shift(1))

# ATR
tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
df['atr_7']    = tr.rolling(7).mean()
df['atr_14']   = tr.rolling(14).mean()
df['atr_ratio']= df['atr_7'] / (df['atr_14'] + 1e-10)

# EMA
ema8  = c.ewm(span=8).mean()
ema21 = c.ewm(span=21).mean()
ema50 = c.ewm(span=50).mean()
df['price_vs_ema8']  = (c - ema8)  / c
df['price_vs_ema21'] = (c - ema21) / c
df['price_vs_ema50'] = (c - ema50) / c
df['ema_slope_8']    = ema8.pct_change(3)
df['ema_cross']      = np.sign(ema8 - ema21) - np.sign(ema8.shift(1) - ema21.shift(1))

# Volume
if df['volume'].sum() > 0:
    df['vol_ratio']  = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
    df['vol_spike']  = (df['vol_ratio'] > 2.0).astype(int)
else:
    df['vol_ratio'] = 1.0
    df['vol_spike'] = 0

# Session
if hasattr(df.index, 'hour'):
    df['hour']       = df.index.hour
    df['minute']     = df.index.minute
else:
    df['hour']   = 0
    df['minute'] = 0
    
df['is_london']  = df['hour'].between(7,  16).astype(int)
df['is_newyork'] = df['hour'].between(13, 21).astype(int)
df['is_overlap'] = df['hour'].between(13, 16).astype(int)
df['session_open'] = (
    ((df['hour'] == 7)  & (df['minute'] < 30)) |
    ((df['hour'] == 13) & (df['minute'] < 30))
).astype(int)

# SMC columns
smc_cols = ['bullish_fvg','bearish_fvg','buy_side_sweep','sell_side_sweep','bullish_mss','bearish_mss']
for col in smc_cols:
    if col in df.columns:
        df[f'{col}_recent'] = df[col].rolling(6).sum()

# ─────────────────────────────────────────────
# 4. GET FEATURE LIST & PREDICT
# ─────────────────────────────────────────────
base = [
    'return_1','return_3','return_6','return_12','return_24',
    'body','upper_wick','lower_wick','hl_range',
    'rsi_7','rsi_14',
    'macd','macd_hist','macd_cross',
    'bb_pos','bb_width',
    'stoch_k','stoch_d','stoch_cross',
    'atr_7','atr_14','atr_ratio',
    'price_vs_ema8','price_vs_ema21','price_vs_ema50',
    'ema_slope_8','ema_cross',
    'vol_ratio','vol_spike',
    'is_london','is_newyork','is_overlap','session_open'
]
smc_extra = [f'{c}_recent' for c in smc_cols if f'{c}_recent' in df.columns]
features = [f for f in base if f in df.columns] + smc_extra

# Get last row data
df_clean = df[features].dropna()
if len(df_clean) == 0:
    print("❌ No valid data after feature engineering")
    exit(1)

X = df_clean.iloc[[-1]]  # Get last row as 2D array
X_sc = scaler.transform(X)

# ─────────────────────────────────────────────
# 5. PREDICT
# ─────────────────────────────────────────────
pred = model.predict(X_sc)[0]
prob = model.predict_proba(X_sc).max(axis=1)[0] * 100

signal_map = {
    1: "BUY",
    0: "HOLD",
    -1: "SELL"
}

print(f"✅ Latest signal (Close: {df['close'].iloc[-1]:.2f}):")
print(f"   Prediction: {signal_map.get(pred, 'UNKNOWN')}")
print(f"   Confidence: {prob:.1f}%")