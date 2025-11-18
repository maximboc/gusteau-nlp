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

    print("JSONL file not found. Converting CSV → instruction format…")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(conversion(row), ensure_ascii=False) + "\n")

    print(f"Saved JSONL dataset → {jsonl_path}")


def conversion(row):
    instruction = (
        "Generate a complete recipe using the following information.\n\n"
        f"Name: {row['name']}\n"
        #f"Ingredients: {row['ingredients_text']}\n"
        # f"Tags: {row['tags_text']}\n"
        # f"Calories: {row['calories']}\n"
        # f"Description: {row['description']}"
    )

    output = row["steps_string_standardize"]
    return {"instruction": instruction, "output": output}
