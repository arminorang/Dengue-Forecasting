import logging
from pathlib import Path
from typing import List
import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class InfoDengueDownloader:
    def __init__(self, base_dir="data/infodengue/raw"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Data directory initialized at: {self.base_dir}")

    def download_municipality_year(self, geocode: int, year: int, disease: str = "dengue") -> pd.DataFrame:
        url = "https://info.dengue.mat.br/api/alertcity"
        params = (
            f"&geocode={geocode}"
            f"&disease={disease}"
            f"&format=csv"
            f"&ew_start=1"
            f"&ew_end=53"
            f"&ey_start={year}"
            f"&ey_end={year}"
        )
        url_resp = "?".join([url, params])

        try:
            df = pd.read_csv(url_resp)
            if df.empty:
                logger.warning(f"No data for {geocode}-{year}")
            return df
        except Exception as e:
            logger.warning(f"Failed to download {geocode}-{year}: {e}")
            return pd.DataFrame()

    def download_multiple(self, geocodes: List[int], years: List[int]):
        """Download multiple municipalities and years, save only one combined CSV"""
        all_data = []
        for geocode in geocodes:
            for year in tqdm(years, desc=f"Downloading geocode {geocode}"):
                df = self.download_municipality_year(geocode, year)
                if not df.empty:
                    df["geocode"] = geocode  # keep track of municipality
                    all_data.append(df)

        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            combined_path = self.base_dir / "infodengue_combined.csv"
            combined.to_csv(combined_path, index=False)
            logger.info(f"Saved combined CSV → {combined_path}")
            return combined
        else:
            logger.warning("No data downloaded")
            return pd.DataFrame()



def main():
    downloader = InfoDengueDownloader()

    # Load all Brazilian municipalities
    url = "https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/csv/municipios.csv"
    df = pd.read_csv(url)

    # Extract only the IBGE codes as integers
    geocodes = df['codigo_ibge'].astype(int).tolist()
    years = list(range(2021, 2025))  # last 20 years

    downloader.download_multiple(geocodes, years)


if __name__ == "__main__":
    main()


