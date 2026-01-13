"""
Simple benchmark to compare generated recipes from fine-tuned models.
Shows the reference recipe and generated output side-by-side.
"""

import random
from datasets import load_dataset
from src.evaluation.judge_llm.judge_llm import RecipeBenchmark, cleanup_resources

def run_recipe_comparison_benchmark(num_samples=3):
    """
    Generate recipes and display them for comparison.
    """
    # Load dataset
    jsonl_path = "data/preprocessed/recipes_instructions.jsonl"
    full_dataset = load_dataset("json", data_files=jsonl_path)["train"]
    
    # Select random samples
    indices = random.sample(range(len(full_dataset)), num_samples)
    test_samples = full_dataset.select(indices)
    
    # Model configurations
    models = [
        {
            "name": "Base Model (No Fine-tuning)",
            "base": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter": None
        },
        {
            "name": "QLoRA Fine-tuned",
            "base": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter": "models/qwen-recipe-qlora"
        },
        {
            "name": "Prompt Tuning Fine-tuned",
            "base": "Qwen/Qwen2.5-0.5B-Instruct",
            "adapter": "models/qwen-recipe-prompt-tuning"
        }
    ]
    
    print("\n" + "="*100)
    print("RECIPE GENERATION BENCHMARK - COMPARING FINE-TUNED MODELS")
    print("="*100)
    
    for idx, sample in enumerate(test_samples):
        print(f"\n{'='*100}")
        print(f"RECIPE #{idx+1}")
        print(f"{'='*100}")
        
        instruction = sample['instruction']
        reference = sample['output']
        
        print(f"\n📝 PROMPT:")
        print(f"{instruction}")
        
        print(f"\n✨ REFERENCE RECIPE:")
        print(f"{'-'*100}")
        print(reference)
        print(f"{'-'*100}")
        
        # Generate with each model
        for model_config in models:
            print(f"\n🤖 {model_config['name'].upper()}:")
            print(f"{'-'*100}")
            
            try:
                benchmark = RecipeBenchmark(
                    model_config['base'],
                    model_config['adapter'],
                    adapter_type=model_config['name']
                )
                
                generated_text, gen_time = benchmark.generate_recipe(instruction)
                
                print(generated_text)
                print(f"\n⏱️  Generation Time: {gen_time:.2f}s")
                print(f"📏 Length: {len(generated_text)} characters")
                print(f"{'-'*100}")
                
                # Cleanup
                del benchmark
                cleanup_resources()
                
            except Exception as e:
                print(f"❌ Error generating recipe: {e}")
                print(f"{'-'*100}")
        
        print("\n" + "="*100 + "\n")
    
    print("\n✅ Benchmark Complete!\n")

if __name__ == "__main__":
    # Run benchmark with 3 sample recipes
    run_recipe_comparison_benchmark(num_samples=3)
