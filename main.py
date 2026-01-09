import random
from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
from src.finetuning.qlora.qlora import qlora_finetuning
from src.evaluation.judge_llm.judge_llm import run_llm_benchmark
from datasets import load_dataset
import os

def main():

    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    adapter_save_path = "models/qwen-recipe-qlora"

    # Preprocessing of the Food.com dataset
    print("--- Step 1: Preprocessing ---")
    data = preprocessing()
    
    print("\n--- Step 2: Loading Dataset ---")
    
    dataset_conversion(data)
    full_dataset = load_dataset("json", data_files=jsonl_path)["train"]
    
    subset_size = 1000
    if len(full_dataset) > subset_size:
        full_dataset = full_dataset.select(range(subset_size))
        
    dataset_dict = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]
    print(f"Data loaded. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Finetuning :
    print("\n--- Step 3: Model Finetuning ---")

    if not os.path.exists(adapter_save_path):
        print("Starting QLoRA training...")
        # We pass the FULL dataset_dict so the trainer has access to ["train"] and ["test"]
        qlora_finetuning("Qwen/Qwen2.5-0.5B-Instruct", dataset_dict, output_dir=adapter_save_path)
    else:
        print(f"Model found at '{adapter_save_path}'. Skipping training. (Delete folder to retrain)")

    """
    # Evaluating :
    print("\n--- Step 4: Benchmarking ---")
    
    random.seed(42) 
    sample_size = min(3, len(test_dataset))
    indices = random.sample(range(len(test_dataset)), sample_size)
    golden_dataset = test_dataset.select(indices)
    
    print(f"✨ Golden Set Created: {len(golden_dataset)} recipes selected.")
    
    competitors = [
        # Config A: The specific LoRA model you just trained
        {
            "name": "Qwen-0.5B (My Recipe LoRA)", 
            "base": "Qwen/Qwen2.5-0.5B-Instruct", 
            "adapter": adapter_save_path 
        },
        # Config B: A larger base model for baseline comparison (Optional)
        # {
        #     "name": "Qwen-1.5B (Baseline)", 
        #     "base": "Qwen/Qwen2.5-1.5B-Instruct", 
        #     "adapter": None 
        # },
    ]

    run_llm_benchmark(
        test_dataset=golden_dataset, 
        model_configs=competitors
    )
    """


if __name__ == "__main__":
    main()
