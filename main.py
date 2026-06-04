from tvDatafeed import Interval
import schedule
import time

from downloader.tv_client import TradingViewDownloader
from downloader.csv_exporter import CSVExporter
from downloader.ict_detector import ICTDetector

def main():

    downloader = TradingViewDownloader()

    df = downloader.download(
        symbol="XAUUSD",
        exchange="OANDA",
        interval=Interval.in_5_minute,
        bars=10000
    )

    print(df.head())

    df = ICTDetector.detect_fvg(df)
    df = ICTDetector.detect_liquidity_sweep(df)
    df = ICTDetector.detect_mss(df)

    CSVExporter.save(df, "xauusd_5m.csv")

if __name__ == "__main__":
    main()




schedule.every(5).minutes.do(main)

while True:
    schedule.run_pending()
    time.sleep(1)