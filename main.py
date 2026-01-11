import random
import os
import torch # Needed to check CUDA availability
from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
from src.finetuning.qlora.qlora import qlora_finetuning
from src.evaluation.judge_llm.judge_llm import run_llm_benchmark
from datasets import load_dataset

def main():
    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    adapter_save_path = "models/qwen-recipe-qlora"

    # --- Step 1: Preprocessing ---
    print("--- Step 1: Preprocessing ---")
    data = preprocessing()
    
    # --- Step 2: Conversion & Advanced Loading ---
    print("\n--- Step 2: Loading & Curating Dataset ---")
    
    # Convert CSV to JSONL
    dataset_conversion(data)
    
    # Load the full dataset
    full_dataset = load_dataset("json", data_files=jsonl_path)["train"]
    print(f"Original Dataset Size: {len(full_dataset)}")

    # 1. QUALITY FILTER: Keep recipes with decent length (e.g., > 200 chars)
    # Short recipes are removed
    print("Filtering for detailed recipes...")
    full_dataset = full_dataset.filter(lambda x: len(x['output']) > 200)
    print(f"Size after filtering short recipes: {len(full_dataset)}")

    # 2. SHUFFLE: Randomize to ensure diversity
    full_dataset = full_dataset.shuffle(seed=42)

    # 3. SELECT SUBSET: 10,000 samples 
    target_size = 10000
    if len(full_dataset) > target_size:
        full_dataset = full_dataset.select(range(target_size))
        
    # Split Train/Test
    dataset_dict = full_dataset.train_test_split(test_size=0.05, seed=42)
    
    print(f"✨ Final Dataset Ready.")
    print(f"Train size: {len(dataset_dict['train'])}")
    print(f"Test size:  {len(dataset_dict['test'])}")
    
    # --- Step 3: Model Finetuning ---
    print("\n--- Step 3: Model Finetuning ---")

    if not os.path.exists(adapter_save_path):
        print(f"Starting QLoRA training on {len(dataset_dict['train'])} examples...")
        
        qlora_finetuning(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            dataset_dict, 
            output_dir=adapter_save_path
        )
    else:
        print(f"Model found at '{adapter_save_path}'. Skipping training.")
    
    # --- Step 4: Benchmarking ---
    print("\n--- Step 4: Benchmarking ---")
    
    # Create a Golden Set from the UNSEEN test data
    random.seed(42) 
    sample_size = min(5, len(dataset_dict['test'])) # Increased to 5 for better sample
    indices = random.sample(range(len(dataset_dict['test'])), sample_size)
    golden_dataset = dataset_dict['test'].select(indices)
    
    competitors = [
        {
            "name": "Qwen-0.5B (Recipe Tuned)", 
            "base": "Qwen/Qwen2.5-0.5B-Instruct", 
            "adapter": adapter_save_path 
        }
    ]

    run_llm_benchmark(
        test_dataset=golden_dataset, 
        model_configs=competitors
    )

if __name__ == "__main__":
    main()
