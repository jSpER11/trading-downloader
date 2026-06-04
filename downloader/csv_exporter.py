import os

class CSVExporter:

    @staticmethod
    def save(df, filename):

        os.makedirs("data/raw", exist_ok=True)

        path = f"data/raw/{filename}"

        df.to_csv(path)

        print(f"[+] Saved: {path}")