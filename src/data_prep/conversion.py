import pandas as pd
import json
import os

jsonl_path = "data/preprocessed/recipes_instructions.jsonl"

def dataset_conversion(df):
    # Ensure parent folder exists
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    # If file already exists → skip preprocessing
    if os.path.exists(jsonl_path):
        print(f"JSONL dataset already exists → {jsonl_path}")
        return

    print("Converting CSV → instruction format…")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            # filter out bad rows before writing
            if not isinstance(row['steps_string_standardize'], str) or len(row['steps_string_standardize']) < 10:
                continue
                
            f.write(json.dumps(conversion(row), ensure_ascii=False) + "\n")


def conversion(row):
    instruction = (
        "Generate a complete recipe using the following information.\n\n"
        f"Name: {row['name']}\n"
        f"Ingredients: {row['ingredients_text']}\n" 
    )

    output = row["steps_string_standardize"]
    return {"instruction": instruction, "output": output}
