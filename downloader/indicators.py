import ta

class Indicators:

    @staticmethod
    def add(df):

        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()

        df["ema_50"] = ta.trend.EMAIndicator(
            df["close"],
            window=50
        ).ema_indicator()

        return df