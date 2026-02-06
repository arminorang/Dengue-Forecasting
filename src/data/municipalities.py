# src/data/download_municipalities.py

import logging
from pathlib import Path
import requests
import zipfile
import io
import geopandas as gpd
import shutil

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class IBGEMunicipalityDownloader:
    def __init__(self, output_dir: str = "data/map"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "brazil_municipalities.gpkg"

        # Correct IBGE 2023 Brazil-wide shapefile URL
        self.shp_url = (
            "https://geoftp.ibge.gov.br/organizacao_do_territorio/"
            "malhas_territoriais/malhas_municipais/municipio_2023/"
            "Brasil/BR_Municipios_2023.zip"
        )

    def download(self) -> gpd.GeoDataFrame:
        logger.info("Downloading IBGE Brazil municipality polygons (2023)...")

        try:
            response = requests.get(self.shp_url, timeout=120)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to download IBGE shapefile: {e}")
            return None

        # Extract ZIP in memory
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            temp_dir = self.output_dir / "tmp_ibge"
            temp_dir.mkdir(exist_ok=True)
            z.extractall(temp_dir)

            # Find the shapefile
            shp_files = list(temp_dir.glob("*.shp"))
            if not shp_files:
                logger.error("No shapefile found in ZIP")
                return None

            shp_path = shp_files[0]
            logger.info(f"Reading shapefile: {shp_path.name}")

            # Read with GeoPandas
            gdf = gpd.read_file(shp_path)

        # Save as GeoPackage
        logger.info(f"Saving GeoPackage to {self.output_file}")
        gdf.to_file(self.output_file, driver="GPKG", layer="municipalities")
        logger.info("Municipality GeoPackage saved successfully!")

        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)

        return gdf


def main():
    downloader = IBGEMunicipalityDownloader()
    downloader.download()


if __name__ == "__main__":
    main()
