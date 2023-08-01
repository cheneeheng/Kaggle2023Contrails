import datetime
import logging
from kaggle.api.kaggle_api_extended import KaggleApi


if __name__ == "__main__":
    api = KaggleApi()
    api.authenticate()
    # # get dataset
    # api.dataset_download_files("eehengchen/chen-contrails-2023",
    #                            path="/tmp",
    #                            unzip=True)
    # upload dataset
    api.dataset_create_version(
        "./kaggle2023contrails",
        dir_mode="zip",
        version_notes=f"Updated on {datetime.datetime.now().strftime('%Y-%m-%d')}",
    )
    logging.info(f"Updated eehengchen/chen-contrails-2023-dummy dataset.")
