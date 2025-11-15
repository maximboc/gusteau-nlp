import os
import pandas as pd
from src.eda.preprocessing import process_data
#from utils.utils_main import download_raw_recipe

RAW_CSV_PATH = './data/RAW_recipes.csv'
PREPROCESSED_CSV_PATH = './data/processed/preprocessed_recipe.csv'

def preprocessing():
    if os.path.isfile(PREPROCESSED_CSV_PATH):
        print("Loading preprocessed data...")
        data = pd.read_csv(PREPROCESSED_CSV_PATH)
    else:
        if not os.path.isfile(RAW_CSV_PATH):
            print("Download manually the dataset from Kaggle (Food.com - RAW_recipes.csv => put in the data/ directory of this project)")
            # download_raw_recipe()
            # print("Preprocessing the dataset...")
            return
        data = process_data(save=True)
        print("Preprocessed data saved and loaded.")

    return data


preprocessing()
