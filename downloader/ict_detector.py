import pandas as pd

class ICTDetector:

    @staticmethod
    def detect_fvg(df):

        df["bullish_fvg"] = (
            (df["low"].shift(-1) > df["high"].shift(1))
        )

        df["bearish_fvg"] = (
            (df["high"].shift(-1) < df["low"].shift(1))
        )

        return df

    @staticmethod
    def detect_liquidity_sweep(df):

        df["buy_side_sweep"] = (
            df["high"] > df["high"].rolling(5).max().shift(1)
        )

        df["sell_side_sweep"] = (
            df["low"] < df["low"].rolling(5).min().shift(1)
        )

        return df

    @staticmethod
    def detect_mss(df):

        df["bullish_mss"] = (
            df["close"] > df["high"].shift(1)
        )

        df["bearish_mss"] = (
            df["close"] < df["low"].shift(1)
        )

        return df