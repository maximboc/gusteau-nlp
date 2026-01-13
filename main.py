import random
import os
import torch # Needed to check CUDA availability
from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
from src.finetuning.qlora.qlora import qlora_finetuning
from src.finetuning.prompt_tuning.prompt_tuning import prompt_tuning_finetuning
from src.evaluation.judge_llm.judge_llm import run_llm_benchmark
from src.evaluation.quantitative.quantitative import run_quantitative_benchmark
from src.evaluation.constrained_generation.outlines_generation import run_constrained_benchmark
from datasets import load_dataset

def main():

    # --- Configuration ---
    # Choose between "qlora" and "prompt_tuning"
    FINETUNING_METHOD = "prompt_tuning" 
    
    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    
    if FINETUNING_METHOD == "qlora":
        adapter_save_path = "models/qwen-recipe-qlora"
    else:
        adapter_save_path = "models/qwen-recipe-prompt-tuning"

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
        print(f"Starting {FINETUNING_METHOD} training...")
        if FINETUNING_METHOD == "qlora":
            qlora_finetuning("Qwen/Qwen2.5-0.5B-Instruct", dataset_dict, output_dir=adapter_save_path)
        elif FINETUNING_METHOD == "prompt_tuning":
            prompt_tuning_finetuning("Qwen/Qwen2.5-0.5B-Instruct", dataset_dict, output_dir=adapter_save_path)
    else:
        print(f"Model found at '{adapter_save_path}'. Skipping training. (Delete folder to retrain)")
    
    
    # Evaluating :
    print("\n--- Step 4: Benchmarking ---")
    
    # Create a Golden Set from the UNSEEN test data
    random.seed(42) 
    sample_size = min(5, len(dataset_dict['test'])) # Increased to 5 for better sample
    indices = random.sample(range(len(dataset_dict['test'])), sample_size)
    golden_dataset = dataset_dict['test'].select(indices)
    
    competitors = [
        # Config A: The specific model you just trained
        {
            "name": f"Qwen-0.5B ({FINETUNING_METHOD})", 
            "base": "Qwen/Qwen2.5-0.5B-Instruct", 
            "adapter": adapter_save_path 
        }
    ]

    run_llm_benchmark(
        test_dataset=golden_dataset, 
        model_configs=competitors
    )

    # 4.2 Quantitative Evaluation
    run_quantitative_benchmark(
         test_dataset=golden_dataset,
         model_configs=competitors
    )

    # 4.3 Outlines Constrained Generation
    run_constrained_benchmark(
        test_dataset=golden_dataset,
        model_configs=competitors
    )


if __name__ == "__main__":
    main()
