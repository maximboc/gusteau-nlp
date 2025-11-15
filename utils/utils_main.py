#from kaggle.api.kaggle_api_extended import KaggleApi
import os

DOWNLOAD_DIR = './data'

def download_raw_recipe():
    """
    api = KaggleApi()
    api.authenticate()

    # Download dataset and unzip
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(
        dataset="shuyangli94/food-com-recipes-and-user-interactions",
        path=DOWNLOAD_DIR,
        unzip=True
    )

    for f in os.listdir(DOWNLOAD_DIR):
        file_path = os.path.join(DOWNLOAD_DIR, f)
        if f != 'RAW_recipes.csv' and os.path.isfile(file_path):
            os.remove(file_path)
    print("Downloaded RAW_recipes.csv and removed other files.")
    """
