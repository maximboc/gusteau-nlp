import random
import os
import torch
from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
from src.finetuning.qlora.qlora import qlora_finetuning
from src.finetuning.ia3.ia3 import ia3_finetuning  # NEW: Import IA³
from src.evaluation.judge_llm.judge_llm import run_llm_benchmark
from datasets import load_dataset

def main():
    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    
    # Define paths for different adapters
    qlora_adapter_path = "models/qwen-recipe-qlora"
    ia3_adapter_path = "models/qwen-recipe-ia3"  # NEW: IA³ adapter path

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
    
    # Choose which fine-tuning method to use
    # Set to "qlora", "ia3", or "both"
    FINETUNING_METHOD = "ia3"  # Change this as needed
    
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"
    
    # QLoRA Fine-tuning
    if FINETUNING_METHOD in ["qlora", "both"]:
        if not os.path.exists(qlora_adapter_path):
            print(f"\n🔥 Starting QLoRA training on {len(dataset_dict['train'])} examples...")
            qlora_finetuning(
                base_model, 
                dataset_dict, 
                output_dir=qlora_adapter_path
            )
        else:
            print(f"✅ QLoRA model found at '{qlora_adapter_path}'. Skipping training.")
    
    # IA³ Fine-tuning (NEW!)
    if FINETUNING_METHOD in ["ia3", "both"]:
        if not os.path.exists(ia3_adapter_path):
            print(f"\n⚡ Starting IA³ training on {len(dataset_dict['train'])} examples...")
            ia3_finetuning(
                base_model,
                dataset_dict,
                output_dir=ia3_adapter_path,
                num_epochs=3,
                batch_size=1,
                gradient_accumulation_steps=8,
                learning_rate=5e-4,
                max_length=512
            )
        else:
            print(f"✅ IA³ model found at '{ia3_adapter_path}'. Skipping training.")

    # --- Step 4: Benchmarking ---
    print("\n--- Step 4: Benchmarking ---")
    
    # Create a Golden Set from the UNSEEN test data
    random.seed(42) 
    sample_size = min(5, len(dataset_dict['test']))
    indices = random.sample(range(len(dataset_dict['test'])), sample_size)
    golden_dataset = dataset_dict['test'].select(indices)
    
    # Define competitors based on what was trained
    competitors = []
    
    # Add base model (no fine-tuning)
    competitors.append({
        "name": "Qwen-0.5B (Base)",
        "base": base_model,
        "adapter": None
    })
    
    # Add QLoRA if available
    if os.path.exists(qlora_adapter_path):
        competitors.append({
            "name": "Qwen-0.5B (QLoRA Tuned)", 
            "base": base_model, 
            "adapter": qlora_adapter_path 
        })
    
    # Add IA³ if available
    if os.path.exists(ia3_adapter_path):
        competitors.append({
            "name": "Qwen-0.5B (IA³ Tuned)",
            "base": base_model,
            "adapter": ia3_adapter_path
        })
    
    print(f"\n🏁 Benchmarking {len(competitors)} models:")
    for comp in competitors:
        print(f"   - {comp['name']}")
    
    run_llm_benchmark(
        test_dataset=golden_dataset, 
        model_configs=competitors
    )
    
    print("\n✅ Benchmarking complete!")

if __name__ == "__main__":
    main()