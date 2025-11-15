# utils/utils_main.py
import os
import pandas as pd
from src.eda.preprocessing import process_data

RAW_CSV_PATH = './data/RAW_recipes.csv'
PREPROCESSED_CSV_PATH = './data/processed/preprocessed_recipe.csv'
DOWNLOAD_DIR = './data'

def download_raw_recipe():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except OSError as e:
        print("❌ Kaggle authentication failed:")
        print(e)
        print("👉 Make sure ~/.kaggle/kaggle.json exists or set KAGGLE_USERNAME & KAGGLE_KEY.")
        return False
    
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(
        dataset="shuyangli94/food-com-recipes-and-user-interactions",
        path=DOWNLOAD_DIR,
        unzip=True,
    )

    for f in os.listdir(DOWNLOAD_DIR):
        file_path = os.path.join(DOWNLOAD_DIR, f)
        if f != "RAW_recipes.csv" and os.path.isfile(file_path):
            os.remove(file_path)

    print("Downloaded RAW_recipes.csv and removed other files.")
    return True


def preprocessing():
    """Download (if needed) and preprocess the dataset."""
    if os.path.isfile(PREPROCESSED_CSV_PATH):
        print("Loading preprocessed data...")
        return pd.read_csv(PREPROCESSED_CSV_PATH)

    if not os.path.isfile(RAW_CSV_PATH):
        print("🔄 RAW_recipes.csv not found. Downloading from Kaggle…")
        success = download_raw_recipe()
        if not success:
            print("\nAlternatively, download the dataset manually. See the README for instructions.")
            return None

    print("Preprocessing the dataset...")
    data = process_data(save=True)
    print("Preprocessed data saved and loaded.")
    return data
