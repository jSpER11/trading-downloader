from tvDatafeed import TvDatafeed, Interval
import pandas as pd

class TradingViewDownloader:

    def __init__(self, username=None, password=None):
        self.tv = TvDatafeed(username, password)

    def download(
        self,
        symbol="XAUUSD",
        exchange="OANDA",
        interval=Interval.in_1_hour,
        bars=5000
    ):

        df = self.tv.get_hist(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            n_bars=bars
        )

        return df