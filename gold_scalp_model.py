"""
Gold Scalp Model — 5M
======================
- Timeframe : 5 минут
- TP        : 0.2%
- SL        : 0.1%
- Hold      : 6-12 candle (30-60 минут)
- Confidence: 75%+ (calibrated)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle, os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TP_PCT               = 0.002   # 0.2%
SL_PCT               = 0.001   # 0.1%
MAX_HOLD             = 12      # 12 candle = 60 минут
CONFIDENCE_THRESHOLD = 75.0
MODEL_PATH           = "gold_5m_model.pkl"
SCALER_PATH          = "gold_5m_scaler.pkl"

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
    sep = ';' if ';' in first_line else ','
    df  = pd.read_csv(filepath, sep=sep)

    date_col = next((c for c in df.columns
                     if 'date' in c.lower() or 'time' in c.lower()), None)
    if date_col:
        sample = str(df[date_col].iloc[0]).strip()
        if sample.lstrip('-').isdigit():
            df[date_col] = pd.to_datetime(df[date_col], unit='s', utc=True)
        else:
            df[date_col] = pd.to_datetime(df[date_col], utc=True, format='ISO8601')
        df[date_col] = df[date_col].dt.tz_localize(None)
        df.set_index(date_col, inplace=True)

    # Symbol багана хасах
    drop = [c for c in df.columns if c.lower() == 'symbol']
    if drop: df = df.drop(columns=drop)

    df.columns = [c.strip().lower() for c in df.columns]

    # Boolean → 0/1
    bool_cols = ['bullish_fvg','bearish_fvg','buy_side_sweep',
                 'sell_side_sweep','bullish_mss','bearish_mss']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda x: 1 if str(x).strip().lower() in ['true','1','yes'] else 0)

    df.sort_index(inplace=True)
    df.dropna(subset=['open','high','low','close'], inplace=True)
    if 'volume' not in df.columns:
        df['volume'] = 0

    print(f"✅ Өгөгдөл: {len(df):,} мөр | {df.index[0]} → {df.index[-1]}")

    # 5M мөрийн тоо шалгах
    expected_5m = (df.index[-1] - df.index[0]).days * 24 * 12
    actual      = len(df)
    coverage    = actual / max(expected_5m, 1) * 100
    print(f"   5M coverage: {coverage:.0f}% (цаг дараалал бүрэн эсэх)")
    return df


# ─────────────────────────────────────────────
# 2. FEATURES — 5M-д тохируулсан
# ─────────────────────────────────────────────
def add_features(df):
    c, h, l, o = df['close'], df['high'], df['low'], df['open']

    # ── Momentum — 5M-д богино window ──
    df['return_1']  = c.pct_change(1)   # 5 мин
    df['return_3']  = c.pct_change(3)   # 15 мин
    df['return_6']  = c.pct_change(6)   # 30 мин
    df['return_12'] = c.pct_change(12)  # 1 цаг
    df['return_24'] = c.pct_change(24)  # 2 цаг

    # ── Candle shape ──
    df['body']       = (c - o) / (c + 1e-10)
    df['upper_wick'] = (h - c.clip(lower=o)) / (h - l + 1e-10)
    df['lower_wick'] = (c.clip(upper=o) - l) / (h - l + 1e-10)
    df['hl_range']   = (h - l) / c

    # ── RSI — 5M-д богино (7, 14) ──
    delta = c.diff()
    for n, name in [(7,'rsi_7'), (14,'rsi_14')]:
        g  = delta.clip(lower=0).rolling(n).mean()
        ls = (-delta.clip(upper=0)).rolling(n).mean()
        df[name] = 100 - (100 / (1 + g / (ls + 1e-10)))

    # ── MACD ──
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    df['macd']       = ema12 - ema26
    df['macd_sig']   = df['macd'].ewm(span=9).mean()
    df['macd_hist']  = df['macd'] - df['macd_sig']
    df['macd_cross'] = np.sign(df['macd_hist']) - np.sign(df['macd_hist'].shift(1))

    # ── Bollinger Bands ──
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df['bb_pos']   = (c - (sma20 - 2*std20)) / (4*std20 + 1e-10)
    df['bb_width'] = 4 * std20 / sma20

    # ── Stochastic ──
    low9  = l.rolling(9).min()
    high9 = h.rolling(9).max()
    df['stoch_k']    = 100 * (c - low9) / (high9 - low9 + 1e-10)
    df['stoch_d']    = df['stoch_k'].rolling(3).mean()
    df['stoch_cross']= np.sign(df['stoch_k'] - df['stoch_d']) - \
                       np.sign(df['stoch_k'].shift(1) - df['stoch_d'].shift(1))

    # ── ATR ──
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    df['atr_7']    = tr.rolling(7).mean()
    df['atr_14']   = tr.rolling(14).mean()
    df['atr_ratio']= df['atr_7'] / (df['atr_14'] + 1e-10)

    # ── EMA — 5M-д богино span ──
    ema8  = c.ewm(span=8).mean()
    ema21 = c.ewm(span=21).mean()
    ema50 = c.ewm(span=50).mean()
    df['price_vs_ema8']  = (c - ema8)  / c
    df['price_vs_ema21'] = (c - ema21) / c
    df['price_vs_ema50'] = (c - ema50) / c
    df['ema_slope_8']    = ema8.pct_change(3)
    df['ema_cross']      = np.sign(ema8 - ema21) - np.sign(ema8.shift(1) - ema21.shift(1))

    # ── Volume ──
    if df['volume'].sum() > 0:
        df['vol_ratio']  = df['volume'] / (df['volume'].rolling(20).mean() + 1e-10)
        df['vol_spike']  = (df['vol_ratio'] > 2.0).astype(int)
    else:
        df['vol_ratio'] = 1.0
        df['vol_spike'] = 0

    # ── Session — 5M-д минут нарийвчлалтай ──
    df['hour']       = df.index.hour
    df['minute']     = df.index.minute
    df['is_london']  = df['hour'].between(7,  16).astype(int)
    df['is_newyork'] = df['hour'].between(13, 21).astype(int)
    df['is_overlap'] = df['hour'].between(13, 16).astype(int)
    # Session эхлэх/дуусах үед volatility өндөр
    df['session_open'] = (
        ((df['hour'] == 7)  & (df['minute'] < 30)) |   # London нээлт
        ((df['hour'] == 13) & (df['minute'] < 30))      # NY нээлт
    ).astype(int)

    # ── ICT/SMC (байгаа бол) ──
    smc_cols = ['bullish_fvg','bearish_fvg','buy_side_sweep',
                'sell_side_sweep','bullish_mss','bearish_mss']
    for col in smc_cols:
        if col in df.columns:
            df[f'{col}_recent'] = df[col].rolling(6).sum()  # 30 мин

    return df


def get_feature_list(df):
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
    smc_extra = [f'{c}_recent' for c in
                 ['bullish_fvg','bearish_fvg','buy_side_sweep',
                  'sell_side_sweep','bullish_mss','bearish_mss']
                 if f'{c}_recent' in df.columns]
    features = [f for f in base if f in df.columns] + smc_extra
    print(f"   Feature тоо: {len(features)} (SMC: {len(smc_extra)})")
    return features


# ─────────────────────────────────────────────
# 3. LABELS
# ─────────────────────────────────────────────
def create_labels(df, tp=TP_PCT, sl=SL_PCT, max_hold=MAX_HOLD):
    labels = np.zeros(len(df), dtype=int)
    closes = df['close'].values
    highs  = df['high'].values
    lows   = df['low'].values

    for i in range(len(df) - max_hold):
        entry    = closes[i]
        tp_price = entry * (1 + tp)
        sl_price = entry * (1 - sl)
        for j in range(1, max_hold + 1):
            if highs[i+j] >= tp_price: labels[i] =  1; break
            if lows[i+j]  <= sl_price: labels[i] = -1; break

    df['label'] = labels
    counts = pd.Series(labels).value_counts().sort_index()
    total  = len(df)
    print(f"\n📊 Label (TP={tp*100:.1f}%, SL={sl*100:.1f}%, max={max_hold} candle={max_hold*5}мин):")
    for lbl, cnt in counts.items():
        name = {1:'Long (TP)', 0:'No trade', -1:'Short (SL)'}[lbl]
        bar  = '█' * int(cnt/total*40)
        print(f"   {name:15s}: {cnt:7,} ({cnt/total*100:.1f}%) {bar}")
    return df


# ─────────────────────────────────────────────
# 4. TRAIN + CALIBRATE
# ─────────────────────────────────────────────
def train_and_calibrate(df, features):
    df_c = df[features + ['label']].dropna()
    X, y = df_c[features], df_c['label']
    n    = len(X)

    i60  = int(n * 0.60)
    i80  = int(n * 0.80)

    X_train, y_train = X.iloc[:i60],    y.iloc[:i60]
    X_cal,   y_cal   = X.iloc[i60:i80], y.iloc[i60:i80]
    X_test,  y_test  = X.iloc[i80:],    y.iloc[i80:]

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_cal_sc   = scaler.transform(X_cal)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n🏋️  Train: {len(X_train):,} | Cal: {len(X_cal):,} | Test: {len(X_test):,}")
    print(f"   ≈ Train: {len(X_train)*5//60//24} өдөр | Test: {len(X_test)*5//60//24} өдөр")

    # Base загвар
    print("   1/3 Base загвар сургаж байна...")
    base = GradientBoostingClassifier(
        n_estimators=300, max_depth=4,
        learning_rate=0.05, subsample=0.8,
        min_samples_leaf=100,  # 5M-д илүү их өгөгдөл тул нэмэгдүүлсэн
        random_state=42
    )
    base.fit(X_train_sc, y_train)

    # Calibration
    print("   2/3 Calibration хийж байна...")

# base-ийн probabilities авах
    cal_proba = base.predict_proba(X_cal_sc)  # (n, 3)

# Шууд sklearn-гүйгээр calibrate хийх
# Эсвэл cv=None, prefit simulation:
    cal_model = CalibratedClassifierCV(
    estimator=base,
    method='isotonic',
    cv=5,
    ensemble=False   # prefit-тэй адил зан үйл
    )
    cal_model.fit(X_train_sc, y_train) 
    # Үнэлгээ
    print("   3/3 Үнэлгээ...")
    y_pred   = cal_model.predict(X_test_sc)
    y_proba  = cal_model.predict_proba(X_test_sc)
    max_conf = y_proba.max(axis=1) * 100

    print(f"\n🎯 Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    print(classification_report(y_test, y_pred,
          target_names=['Short','No trade','Long'], zero_division=0))

    # Threshold хүснэгт
    print(f"\n📊 Confidence threshold шүүлт:")
    print(f"   {'Threshold':>10} | {'Trade тоо':>10} | {'Win rate':>9} | {'Avg conf':>9} | {'Өдөрт':>7}")
    print(f"   {'-'*10}-+-{'-'*10}-+-{'-'*9}-+-{'-'*9}-+-{'-'*7}")
    test_days = len(X_test) * 5 / 60 / 24
    for thr in [50, 55, 60, 65, 70, 75, 80, 85]:

        mask = max_conf >= thr
    
        if mask.sum() == 0:
            continue
        
        yp = pd.Series(y_pred[mask]).reset_index(drop=True)
        yt = pd.Series(y_test[mask]).reset_index(drop=True)
    
        nz_mask = yp != 0
    
        nz    = yp[nz_mask]
        yt_nz = yt[nz_mask]
    
        wr = (nz.values == yt_nz.values).mean() * 100 if len(nz) > 0 else 0
    
        avg_c = max_conf[mask].mean()
        per_day = mask.sum() / max(test_days, 1)
    
        print(
            f"{thr:>3}% | "
            f"signals={mask.sum():>6,} | "
            f"winrate={wr:>6.1f}% | "
            f"avg_conf={avg_c:>6.1f}% | "
            f"{per_day:>5.1f}/day"
        )
    # Хадгалах
    with open(MODEL_PATH,  'wb') as f: pickle.dump(cal_model, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler,    f)
    print(f"\n💾 {MODEL_PATH} хадгалагдлаа")

    return cal_model, scaler, X_test, y_test, y_pred, max_conf, df_c.index[i80:]


# ─────────────────────────────────────────────
# 5. BACKTEST
# ─────────────────────────────────────────────
def backtest(df, test_index, y_pred, max_conf,
             tp=TP_PCT, sl=SL_PCT, max_hold=MAX_HOLD,
             capital=10000, spread=0.0001,
             conf_thr=CONFIDENCE_THRESHOLD):

    prices  = df[['open','high','low','close']].loc[test_index]
    signals = pd.Series(y_pred,   index=test_index)
    confs   = pd.Series(max_conf, index=test_index)

    equity  = capital
    curve   = []
    trades  = []
    skipped = 0
    in_trade= False
    entry_p = direction = entry_i = 0

    for i, idx in enumerate(prices.index):
        row = prices.loc[idx]

        if in_trade:
            held   = i - entry_i
            hit_tp = (direction== 1 and row['high'] >= entry_p*(1+tp)) or \
                     (direction==-1 and row['low']  <= entry_p*(1-tp))
            hit_sl = (direction== 1 and row['low']  <= entry_p*(1-sl)) or \
                     (direction==-1 and row['high'] >= entry_p*(1+sl))
            if hit_tp:
                equity *= (1 + tp - spread)
                trades.append({'idx':idx,'result':'TP','conf':confs.iloc[i],'hold':held})
                in_trade = False
            elif hit_sl or held >= max_hold:
                pnl = -sl-spread if hit_sl else \
                      (row['close']/entry_p-1)*direction - spread
                equity *= (1 + pnl)
                trades.append({'idx':idx,'result':'SL' if hit_sl else 'Timeout',
                                'conf':confs.iloc[i],'hold':held})
                in_trade = False

        if not in_trade and i < len(prices) - max_hold:
            sig, conf = signals.iloc[i], confs.iloc[i]
            if sig in [1,-1]:
                if conf >= conf_thr:
                    in_trade = True; entry_p = row['close']
                    entry_i  = i;    direction = sig
                else:
                    skipped += 1

        curve.append(equity)

    curve_s   = pd.Series(curve, index=test_index)
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    n         = len(trades_df)
    wr        = (trades_df['result']=='TP').mean()*100 if n>0 else 0
    total_ret = (equity/capital - 1)*100
    bh_ret    = (prices['close'].iloc[-1]/prices['close'].iloc[0]-1)*100
    max_dd    = ((curve_s/curve_s.cummax())-1).min()*100
    dr        = curve_s.resample('D').last().pct_change().dropna()
    sharpe    = (dr.mean()/dr.std())*np.sqrt(252) if dr.std()>0 else 0
    test_days = len(test_index)*5/60/24

    print(f"\n{'='*55}")
    print(f"  BACKTEST — 5M  |  Conf ≥ {conf_thr:.0f}%")
    print(f"{'='*55}")
    print(f"  Strategy Return  : {total_ret:+.1f}%")
    print(f"  Buy & Hold       : {bh_ret:+.1f}%")
    print(f"  Max Drawdown     : {max_dd:.1f}%")
    print(f"  Sharpe Ratio     : {sharpe:.2f}")
    print(f"  Нийт Trade       : {n:,}  (~{n/max(test_days,1):.1f}/өдөр)")
    print(f"  Алгасан signal   : {skipped:,} (conf < {conf_thr:.0f}%)")
    print(f"  Win Rate (TP)    : {wr:.1f}%")
    if n > 0:
        tp_n  = (trades_df['result']=='TP').sum()
        sl_n  = (trades_df['result']=='SL').sum()
        to_n  = (trades_df['result']=='Timeout').sum()
        avg_h = trades_df['hold'].mean() * 5
        avg_c = trades_df['conf'].mean()
        print(f"  TP/SL/Timeout    : {tp_n}/{sl_n}/{to_n}")
        print(f"  Avg hold         : {avg_h:.0f} мин")
        print(f"  Avg confidence   : {avg_c:.1f}%")
    print(f"{'='*55}")

    # График
    _plot(prices, curve_s, trades_df)
    return curve_s, trades_df


def _plot(prices, curve_s, trades_df):
    fig, axes = plt.subplots(3, 1, figsize=(15,10), facecolor='#0d0d0d')
    fig.suptitle('Gold Scalp 5M — Calibrated', color='gold', fontsize=14, fontweight='bold')

    for ax in axes:
        ax.set_facecolor('#111')
        ax.tick_params(colors='#aaa')
        for sp in ax.spines.values(): sp.set_color('#333')

    c = prices['close']
    axes[0].plot(c.index, c.values, color='#666', linewidth=0.5)
    if len(trades_df):
        tp_t = trades_df[trades_df['result']=='TP']
        sl_t = trades_df[trades_df['result']=='SL']
        to_t = trades_df[trades_df['result']=='Timeout']
        if len(tp_t): axes[0].scatter(tp_t['idx'], c.loc[tp_t['idx']], marker='^', color='#00ff88', s=30, zorder=5, label=f'TP ({len(tp_t)})')
        if len(sl_t): axes[0].scatter(sl_t['idx'], c.loc[sl_t['idx']], marker='v', color='#ff4455', s=30, zorder=5, label=f'SL ({len(sl_t)})')
        if len(to_t): axes[0].scatter(to_t['idx'], c.loc[to_t['idx']], marker='o', color='#ffaa00', s=15, zorder=5, label=f'Timeout ({len(to_t)})', alpha=0.5)
    axes[0].legend(facecolor='#111', labelcolor='white', fontsize=8)
    axes[0].set_title('Үнэ & Trade-ууд', color='white')

    bh = (c / c.iloc[0]) * curve_s.iloc[0]
    axes[1].plot(curve_s.index, curve_s.values, color='#00aaff', linewidth=1.5, label='Scalp 5M')
    axes[1].plot(bh.index, bh.values, color='#555', linewidth=1, linestyle='--', label='Buy & Hold')
    axes[1].fill_between(curve_s.index, curve_s.values, curve_s.iloc[0], alpha=0.1, color='#00aaff')
    axes[1].legend(facecolor='#111', labelcolor='white', fontsize=8)
    axes[1].set_title('Equity Curve', color='white')

    dd = (curve_s / curve_s.cummax() - 1) * 100
    axes[2].fill_between(dd.index, dd.values, 0, color='#ff4455', alpha=0.4)
    axes[2].plot(dd.index, dd.values, color='#ff4455', linewidth=0.8)
    axes[2].set_title('Drawdown %', color='white')
    axes[2].set_ylabel('DD %', color='#aaa')

    plt.tight_layout()
    plt.savefig('gold_5m_results.png', dpi=130, bbox_inches='tight', facecolor='#0d0d0d')
    plt.show()
    print("📊 gold_5m_results.png хадгалагдлаа")


# ─────────────────────────────────────────────
# 6. SIGNAL
# ─────────────────────────────────────────────
def get_signal(df, model, scaler, features):
    df2  = add_features(df.copy())
    last = df2[features].dropna().iloc[-1:]
    if last.empty: return

    X_sc  = scaler.transform(last)
    pred  = model.predict(X_sc)[0]
    proba = model.predict_proba(X_sc)[0]
    conf  = proba[list(model.classes_).index(pred)] * 100
    price = df['close'].iloc[-1]
    row   = df2.iloc[-1]

    smc = [col for col in ['bullish_fvg','bearish_fvg','buy_side_sweep',
                            'sell_side_sweep','bullish_mss','bearish_mss']
           if col in df.columns and df[col].iloc[-1] == 1]

    print(f"\n{'='*52}")
    if pred == 0 or conf < CONFIDENCE_THRESHOLD:
        msg = f'⚪ SKIP — conf {conf:.1f}%' if pred != 0 else '⚪ NO TRADE'
        print(f"  {msg}")
    else:
        print(f"  {'🟢 LONG' if pred==1 else '🔴 SHORT'}  ✅  conf {conf:.1f}%")
        print(f"{'='*52}")
        print(f"  Үнэ       : ${price:,.2f}")
        print(f"  TP        : ${price*(1+TP_PCT*(1 if pred==1 else -1)):,.2f}  (+{TP_PCT*100:.1f}%)")
        print(f"  SL        : ${price*(1-SL_PCT*(1 if pred==1 else -1)):,.2f}  (-{SL_PCT*100:.1f}%)")
        print(f"  RSI-14    : {row.get('rsi_14', 0):.1f}")
        print(f"  BB позиц  : {row.get('bb_pos', 0):.2f}")
        print(f"  ATR       : {row.get('atr_7', 0):.2f}")
        tp_min = MAX_HOLD * 5
        print(f"  Max hold  : {tp_min} мин ({MAX_HOLD} candle)")
    if smc:
        print(f"  SMC       : {', '.join(smc)}")
    print(f"{'='*52}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    FILEPATH = "data/raw/xauusd_5m.csv"   # ✏️ Өөрийн 5M файл

    print("="*55)
    print(f"  GOLD SCALP 5M  |  TP=0.2%  SL=0.1%  Hold≤60мин")
    print(f"  Confidence ≥ {CONFIDENCE_THRESHOLD:.0f}%  |  Calibrated")
    print("="*55)

    df       = load_data(FILEPATH)
    df       = add_features(df)
    features = get_feature_list(df)
    df       = create_labels(df)

    model, scaler, X_test, y_test, y_pred, max_conf, test_index = \
        train_and_calibrate(df, features)

    equity, trades_df = backtest(df, test_index, y_pred, max_conf)

    print("\n📡 Сүүлийн 5M candle-н signal:")
    get_signal(df, model, scaler, features)

    print("\n✅ Дууслаа!")