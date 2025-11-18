from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
import pandas as pd
import json
from datasets import load_dataset

jsonl_path = "data/preprocessed/recipes_instructions.jsonl"

if __name__ == "__main__":
    print("Preprocessing")
    # Preprocessing of the Food.com dataset
    data = preprocessing()
    
    print("Converting the dataset for model fine tuning")
    dataset_conversion(data)
    dataset = load_dataset("json", data_files=jsonl_path)["train"]
    # dataset = dataset.train_test_split(test_size=0.05)

    # Finetuning :
    print("Starting the different qwen-05b finetuning process...")
    print("1. QLORA finetuning process...")
