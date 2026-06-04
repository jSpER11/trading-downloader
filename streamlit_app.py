import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAUUSD Gold Dashboard",
    page_icon="🥇",
    layout="wide",
)

st.markdown("""
<style>
    .signal-buy  { background:#0d3320; color:#00ff88; border:1px solid #00ff88;
                   border-radius:8px; padding:20px; text-align:center; font-size:2rem; font-weight:bold; }
    .signal-sell { background:#3d0d0d; color:#ff4444; border:1px solid #ff4444;
                   border-radius:8px; padding:20px; text-align:center; font-size:2rem; font-weight:bold; }
    .signal-hold { background:#1a1a2e; color:#aaaaaa; border:1px solid #555;
                   border-radius:8px; padding:20px; text-align:center; font-size:2rem; font-weight:bold; }
    .metric-card { background:#0f0f1a; border:1px solid #222; border-radius:8px; padding:16px; }
</style>
""", unsafe_allow_html=True)

st.title("🥇 XAUUSD Gold Scalping Dashboard")

# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Тохиргоо")
    bars = st.slider("Мөрийн тоо (bars)", 500, 10000, 2000, step=500)

    interval_map = {
        "5 минут":  "in_5_minute",
        "15 минут": "in_15_minute",
        "1 цаг":    "in_1_hour",
    }
    interval_label = st.selectbox("Timeframe", list(interval_map.keys()))

    symbol   = st.text_input("Symbol",   value="XAUUSD")
    exchange = st.text_input("Exchange", value="OANDA")

    model_path  = st.text_input("Model файл",  value="gold_5m_model.pkl")
    scaler_path = st.text_input("Scaler файл", value="gold_5m_scaler.pkl")

    run_btn = st.button("🚀 Ажиллуулах", use_container_width=True)

# ── Helper: feature engineering (gold_scalp_signal.py -аас авсан) ─────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Boolean ICT columns → int
    bool_cols = ['bullish_fvg','bearish_fvg','buy_side_sweep',
                 'sell_side_sweep','bullish_mss','bearish_mss']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: 1 if str(x).strip().lower() in ['true','1','yes'] else 0
            )

    c, h, l, o = df['close'], df['high'], df['low'], df['open']

    # Momentum
    for n in [1, 3, 6, 12, 24]:
        df[f'return_{n}'] = c.pct_change(n)

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
    df['macd_cross'] = (np.sign(df['macd_hist'])
                        - np.sign(df['macd_hist'].shift(1)))

    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_pos']   = (c - (sma20 - 2*std20)) / (4*std20 + 1e-10)
    df['bb_width'] = 4 * std20 / sma20

    # Stochastic
    low9  = l.rolling(9).min()
    high9 = h.rolling(9).max()
    df['stoch_k']     = 100 * (c - low9) / (high9 - low9 + 1e-10)
    df['stoch_d']     = df['stoch_k'].rolling(3).mean()
    df['stoch_cross'] = (np.sign(df['stoch_k'] - df['stoch_d'])
                         - np.sign(df['stoch_k'].shift(1) - df['stoch_d'].shift(1)))

    # ATR
    tr = pd.concat([h - l,
                    (h - c.shift()).abs(),
                    (l - c.shift()).abs()], axis=1).max(axis=1)
    df['atr_7']     = tr.rolling(7).mean()
    df['atr_14']    = tr.rolling(14).mean()
    df['atr_ratio'] = df['atr_7'] / (df['atr_14'] + 1e-10)

    # EMA
    ema8  = c.ewm(span=8).mean()
    ema21 = c.ewm(span=21).mean()
    ema50 = c.ewm(span=50).mean()
    df['price_vs_ema8']  = (c - ema8)  / c
    df['price_vs_ema21'] = (c - ema21) / c
    df['price_vs_ema50'] = (c - ema50) / c
    df['ema_slope_8']    = ema8.pct_change(3)
    df['ema_cross']      = (np.sign(ema8 - ema21)
                            - np.sign(ema8.shift(1) - ema21.shift(1)))

    # Volume
    if 'volume' in df.columns and df['volume'].sum() > 0:
        df['vol_ratio'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['vol_spike'] = (df['vol_ratio'] > 2.0).astype(int)
    else:
        df['vol_ratio'] = 1.0
        df['vol_spike'] = 0

    # Session
    if hasattr(df.index, 'hour'):
        df['hour']   = df.index.hour
        df['minute'] = df.index.minute
    else:
        df['hour']   = 0
        df['minute'] = 0

    df['is_london']    = df['hour'].between(7, 16).astype(int)
    df['is_newyork']   = df['hour'].between(13, 21).astype(int)
    df['is_overlap']   = df['hour'].between(13, 16).astype(int)
    df['session_open'] = (
        ((df['hour'] == 7)  & (df['minute'] < 30)) |
        ((df['hour'] == 13) & (df['minute'] < 30))
    ).astype(int)

    # SMC rolling counts
    smc_cols = ['bullish_fvg','bearish_fvg','buy_side_sweep',
                'sell_side_sweep','bullish_mss','bearish_mss']
    for col in smc_cols:
        if col in df.columns:
            df[f'{col}_recent'] = df[col].rolling(6).sum()

    return df


FEATURE_LIST = [
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
    'is_london','is_newyork','is_overlap','session_open',
    'bullish_fvg_recent','bearish_fvg_recent',
    'buy_side_sweep_recent','sell_side_sweep_recent',
    'bullish_mss_recent','bearish_mss_recent',
]

SIGNAL_MAP = {1: "BUY 🟢", 0: "HOLD ⚪", -1: "SELL 🔴"}


# ── Main logic ──────────────────────────────────────────────────────────────────
if run_btn:
    # 1. TradingView-ээс өгөгдөл татах
    try:
        from tvDatafeed import TvDatafeed, Interval
        from downloader.tv_client import TradingViewDownloader
        from downloader.ict_detector import ICTDetector

        interval_attr = interval_map[interval_label]

        with st.spinner("📡 TradingView-ээс өгөгдөл татаж байна..."):
            downloader = TradingViewDownloader()
            df_raw = downloader.download(
                symbol=symbol,
                exchange=exchange,
                interval=getattr(Interval, interval_attr),
                bars=bars,
            )
            df_raw = ICTDetector.detect_fvg(df_raw)
            df_raw = ICTDetector.detect_liquidity_sweep(df_raw)
            df_raw = ICTDetector.detect_mss(df_raw)
            st.session_state['df_raw'] = df_raw
            st.success(f"✅ {len(df_raw)} мөр татаж авлаа")

    except Exception as e:
        st.error(f"❌ Өгөгдөл татахад алдаа гарлаа: {e}")
        st.stop()

    # 2. Feature engineering
    with st.spinner("🔧 Feature engineering..."):
        df_feat = add_features(df_raw)

    # 3. ML сигнал
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model  = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            features = [f for f in FEATURE_LIST if f in df_feat.columns]
            df_clean = df_feat[features].dropna()

            if len(df_clean) == 0:
                st.warning("⚠️ Feature engineering-ийн дараа хангалттай өгөгдөл алга.")
            else:
                X_last   = df_clean.iloc[[-1]]
                X_scaled = scaler.transform(X_last)

                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled).max(axis=1)[0] * 100

                st.session_state['pred'] = pred
                st.session_state['prob'] = prob
                st.session_state['close'] = df_feat['close'].iloc[-1]

                # Batch predict for history
                X_all    = df_clean
                X_all_sc = scaler.transform(X_all)
                df_feat.loc[df_clean.index, 'signal'] = model.predict(X_all_sc)

        except Exception as e:
            st.error(f"❌ Моделийн алдаа: {e}")
    else:
        st.warning(f"⚠️ Model файл олдсонгүй: `{model_path}` эсвэл `{scaler_path}`\n\n"
                   "Сигнал харуулахгүй, зөвхөн өгөгдлийн хүснэгтийг харуулна.")

    st.session_state['df_feat'] = df_feat


# ── Display ─────────────────────────────────────────────────────────────────────

# ── TradingView Chart ───────────────────────────────────────────────────────────
st.subheader("📊 XAUUSD Live Chart")

tv_interval_map = {
    "5 минут":  "5",
    "15 минут": "15",
    "1 цаг":    "60",
}
tv_interval = tv_interval_map.get(interval_label, "5")

tradingview_widget = f"""
<div class="tradingview-widget-container" style="height:520px;">
  <div id="tradingview_chart" style="height:500px;"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
  <script type="text/javascript">
    new TradingView.widget({{
      "autosize": true,
      "symbol": "OANDA:XAUUSD",
      "interval": "{tv_interval}",
      "timezone": "Asia/Ulaanbaatar",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": "#0f0f1a",
      "enable_publishing": false,
      "hide_side_toolbar": false,
      "allow_symbol_change": true,
      "studies": [
        "RSI@tv-basicstudies",
        "MACD@tv-basicstudies"
      ],
      "container_id": "tradingview_chart"
    }});
  </script>
</div>
"""
st.components.v1.html(tradingview_widget, height=520)

st.divider()

if 'df_feat' in st.session_state:
    df_feat = st.session_state['df_feat']
    df_raw  = st.session_state.get('df_raw', df_feat)

    # ── Сигнал хэсэг ──
    if 'pred' in st.session_state:
        pred  = st.session_state['pred']
        prob  = st.session_state['prob']
        close = st.session_state['close']

        st.subheader("🤖 ML Сигнал")
        col_sig, col_price, col_conf = st.columns(3)

        signal_label = SIGNAL_MAP.get(pred, "UNKNOWN")
        css_class = {1: "signal-buy", 0: "signal-hold", -1: "signal-sell"}.get(pred, "signal-hold")

        with col_sig:
            st.markdown(f'<div class="{css_class}">{signal_label}</div>', unsafe_allow_html=True)
        with col_price:
            st.metric("Сүүлийн үнэ", f"${close:,.2f}")
        with col_conf:
            st.metric("Confidence", f"{prob:.1f}%")

        st.divider()

    # ── ICT метрик ──
    st.subheader("📊 ICT Indicator үзүүлэлт")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    ict_map = {
        'bullish_fvg':     (m1, "🟢 Bull FVG"),
        'bearish_fvg':     (m2, "🔴 Bear FVG"),
        'buy_side_sweep':  (m3, "💧 Buy Sweep"),
        'sell_side_sweep': (m4, "💧 Sell Sweep"),
        'bullish_mss':     (m5, "🔼 Bull MSS"),
        'bearish_mss':     (m6, "🔽 Bear MSS"),
    }
    for col_name, (col_widget, label) in ict_map.items():
        if col_name in df_raw.columns:
            col_widget.metric(label, int(df_raw[col_name].sum()))
        else:
            col_widget.metric(label, "N/A")

    st.divider()

    # ── Indicator хүснэгт ──
    st.subheader("📈 Сүүлийн 100 мөрийн indicator")
    show_cols = ['close', 'rsi_14', 'macd_hist', 'bb_pos',
                 'stoch_k', 'atr_14', 'vol_ratio']
    show_cols = [c for c in show_cols if c in df_feat.columns]
    if 'signal' in df_feat.columns:
        show_cols = ['signal'] + show_cols

    def color_signal(val):
        if val == 1:  return 'background-color:#0d3320; color:#00ff88'
        if val == -1: return 'background-color:#3d0d0d; color:#ff4444'
        return ''

    display_df = df_feat[show_cols].dropna().tail(100)
    if 'signal' in display_df.columns:
        try:
            styled = display_df.style.map(color_signal, subset=['signal'])
        except AttributeError:
            styled = display_df.style.applymap(color_signal, subset=['signal'])
        st.dataframe(styled, use_container_width=True, height=400)
    else:
        st.dataframe(display_df, use_container_width=True, height=400)

    st.divider()

    # ── Хэрхэн уншихаа CSV татах ──
    st.subheader("⬇️ Өгөгдөл татах")
    c1, c2 = st.columns(2)
    with c1:
        csv_raw = df_raw.to_csv().encode('utf-8')
        st.download_button("📥 Raw өгөгдөл (CSV)",
                           csv_raw, "xauusd_raw.csv", "text/csv",
                           use_container_width=True)
    with c2:
        csv_feat = df_feat.to_csv().encode('utf-8')
        st.download_button("📥 Feature өгөгдөл (CSV)",
                           csv_feat, "xauusd_features.csv", "text/csv",
                           use_container_width=True)

else:
    st.info("👈 Зүүн талын **Ажиллуулах** товчийг дарж эхлүүлнэ үү.")
