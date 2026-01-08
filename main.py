import random
from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
from src.finetuning.qlora.qlora import qlora_finetuning
from src.finetuning.prompt_tuning.prompt_tuning import prompt_tuning_finetuning
from src.evaluation.judge_llm.judge_llm import run_llm_benchmark
from src.evaluation.quantitative.quantitative import run_quantitative_benchmark
from datasets import load_dataset
import os

def main():

    # --- Configuration ---
    # Choose between "qlora" and "prompt_tuning"
    FINETUNING_METHOD = "prompt_tuning" 
    
    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    
    if FINETUNING_METHOD == "qlora":
        adapter_save_path = "models/qwen-recipe-qlora"
    else:
        adapter_save_path = "models/qwen-recipe-prompt-tuning"

    # Preprocessing of the Food.com dataset
    print("--- Step 1: Preprocessing ---")
    data = preprocessing()
    
    print("\n--- Step 2: Loading Dataset ---")
    
    dataset_conversion(data)
    full_dataset = load_dataset("json", data_files=jsonl_path)["train"]
    dataset_dict = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = dataset_dict["train"]
    test_dataset = dataset_dict["test"]
    print(f"Data loaded. Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    # Finetuning :
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
    
    random.seed(42) 
    sample_size = min(3, len(test_dataset))
    indices = random.sample(range(len(test_dataset)), sample_size)
    golden_dataset = test_dataset.select(indices)
    
    print(f"✨ Golden Set Created: {len(golden_dataset)} recipes selected.")
    
    competitors = [
        # Config A: The specific model you just trained
        {
            "name": f"Qwen-0.5B ({FINETUNING_METHOD})", 
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

    # 4.2 Quantitative Evaluation
    run_quantitative_benchmark(
         test_dataset=golden_dataset,
         model_configs=competitors
    )


if __name__ == "__main__":
    main()
