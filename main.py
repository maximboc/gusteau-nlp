import random
import os
import torch
from src.utils.utils_main import preprocessing
from src.data_prep.conversion import dataset_conversion
from src.finetuning.qlora.qlora import qlora_finetuning
from src.finetuning.prompt_tuning.prompt_tuning import prompt_tuning_finetuning
from src.finetuning.ia3.ia3 import ia3_finetuning
from src.evaluation.judge_llm.judge_llm import run_llm_benchmark
from src.evaluation.quantitative.quantitative import run_quantitative_benchmark
from src.evaluation.constrained_generation.outlines_generation import run_constrained_benchmark
from src.dspy_optimization.dspy_optimizer import check_dspy_optimization, run_dspy_benchmark
from src.evaluation.qualitative.showcase import run_qualitative_showcase
from datasets import load_dataset

def main():

    # --- Configuration ---
    # Choose between "qlora", "ia3" and "prompt_tuning"
    FINETUNING_METHOD = "prompt_tuning" 
    ENABLE_DSPY = False # Enable DSPy Prompt Optimization
    
    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    
    if FINETUNING_METHOD == "qlora":
        adapter_save_path = "models/qwen-recipe-qlora2"
    elif FINETUNING_METHOD == "prompt_tuning":
        adapter_save_path = "models/qwen-recipe-prompt-tuning"
    else:
        adapter_save_path = "models/qwen-recipe-ia3"

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
    
    # Check if adapter_config.json exists (not just the directory)
    adapter_config_path = os.path.join(adapter_save_path, "adapter_config.json")
    
    if not os.path.exists(adapter_config_path):
        print(f"Starting {FINETUNING_METHOD} training...")
        if FINETUNING_METHOD == "qlora":
            qlora_finetuning("Qwen/Qwen2.5-0.5B-Instruct", dataset_dict, output_dir=adapter_save_path)
        elif FINETUNING_METHOD == "prompt_tuning":
            prompt_tuning_finetuning("Qwen/Qwen2.5-0.5B-Instruct", dataset_dict, output_dir=adapter_save_path)
        elif FINETUNING_METHOD == "ia3":
            ia3_finetuning("Qwen/Qwen2.5-0.5B-Instruct", dataset_dict, output_dir=adapter_save_path,
                num_epochs=3, batch_size=1, gradient_accumulation_steps=8, learning_rate=5e-4, max_length=512)
    else:
        print(f"Trained adapter found at '{adapter_save_path}'. Skipping training. (Delete to retrain)")
    
    
    # Evaluating :
    print("\n--- Step 4: Benchmarking ---")
    
    # Create a Golden Set from the UNSEEN test data
    random.seed(42) 
    sample_size = min(5, len(dataset_dict['test']))
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
    if FINETUNING_METHOD != "ia3":
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
    
    print("\n✅ Benchmarking complete!")

    # 4.4 DSPy Optimization & Generation
    if ENABLE_DSPY:
        print("\n--- Step 5: DSPy Prompt Optimization ---")
        # Use the same HuggingFace model we're using for finetuning
        dspy_program, dspy_lm = check_dspy_optimization(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            dataset_dict,
            adapter_path=adapter_save_path
        )
        
        if dspy_program is not None and dspy_lm is not None:
            # Generate with DSPy (passing custom_lm for local models)
            dspy_results = run_dspy_benchmark(golden_dataset, dspy_program, custom_lm=dspy_lm)
            
            # Show sample output
            if dspy_results:
                print("\n📝 Sample DSPy Generation:")
                sample = dspy_results[0]
                print(f"Dish: {sample['dish'][:60]}...")
                print(f"Generated Recipe Preview:")
                print(sample['generated_recipe'][:300] + "..." if len(sample['generated_recipe']) > 300 else sample['generated_recipe'])
                print("-" * 60)
        else:
            print("⚠️ DSPy optimization skipped or failed")
    else:
        print("\n--- Step 5: DSPy Optimization ---")
        print("⚠️ DSPy disabled in configuration (ENABLE_DSPY=False)")

    # 4.5 Qualitative Showcase
    print("\n--- Step 6: Qualitative Showcase ---")
    run_qualitative_showcase(
        dataset=golden_dataset,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=adapter_save_path
    )
    


    # 4.4 DSPy Optimization & Generation
    if ENABLE_DSPY:
        print("\n--- Step 5: DSPy Prompt Optimization ---")
        # Use the same HuggingFace model we're using for finetuning
        dspy_program, dspy_lm = check_dspy_optimization(
            "Qwen/Qwen2.5-0.5B-Instruct", 
            dataset_dict,
            adapter_path=adapter_save_path
        )
        
        if dspy_program is not None and dspy_lm is not None:
            # Generate with DSPy (passing custom_lm for local models)
            dspy_results = run_dspy_benchmark(golden_dataset, dspy_program, custom_lm=dspy_lm)
            
            # Show sample output
            if dspy_results:
                print("\n📝 Sample DSPy Generation:")
                sample = dspy_results[0]
                print(f"Dish: {sample['dish'][:60]}...")
                print(f"Generated Recipe Preview:")
                print(sample['generated_recipe'][:300] + "..." if len(sample['generated_recipe']) > 300 else sample['generated_recipe'])
                print("-" * 60)
        else:
            print("⚠️ DSPy optimization skipped or failed")
    else:
        print("\n--- Step 5: DSPy Optimization ---")
        print("⚠️ DSPy disabled in configuration (ENABLE_DSPY=False)")

    # 4.5 Qualitative Showcase
    print("\n--- Step 6: Qualitative Showcase ---")
    run_qualitative_showcase(
        dataset=golden_dataset,
        base_model_id="Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path=adapter_save_path
    )

if __name__ == "__main__":
    main()
