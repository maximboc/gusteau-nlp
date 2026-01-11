import time
import torch
import gc
import os
import pandas as pd
import random
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from src.utils.utils import get_device
from src.evaluation.judge_llm.utils.gemini import setup_gemini, judge_recipe

@dataclass
class BenchmarkResult:
    model_name: str
    dish_name: str
    generation_time: float
    score: int
    reasoning: str
    recipe_length: int
    judge_model: str

def cleanup_resources():
    """
    Frees memory on NVIDIA GPUs, Apple MPS (Mac), or CPU.
    Crucial when switching between models to avoid OOM errors.
    """
    # 1. Force Python's Garbage Collector
    gc.collect()

    # 2. Clear NVIDIA Cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    # 3. Clear Apple MPS (Mac) Cache
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except AttributeError:
            pass

class RecipeBenchmark:
    def __init__(self, base_model_id: str, adapter_path: Optional[str] = None):
        self.device_kwargs, _ = get_device()
        print(f"🏗️ Initializing Benchmark with Base: {base_model_id}")
        
        # 1. Load Base Model
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id, **self.device_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Failed to load base model {base_model_id}: {e}")
        
        # 2. Load Adapter (if provided, exists, and is valid)
        self.has_adapter = False
        self.model = self.base_model # Default to base

        if adapter_path:
            if os.path.exists(adapter_path):
                print(f"🔗 Loading Adapter from: {adapter_path}")
                try:
                    self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
                    self.has_adapter = True
                except Exception as e:
                    print(f"❌ Error loading adapter (Fallback to Base): {e}")
            else:
                print(f"⚠️ WARNING: Adapter path '{adapter_path}' not found.")
                print("   -> Proceeding with Base Model only.")

    def extract_dish_name(self, instruction_text: str) -> str:
        """Parses the instruction string to find the dish name."""
        try:
            # Matches "Create a detailed recipe for {Dish Name}."
            match = re.search(r"recipe for\s*(.+?)(?:\.|$)", instruction_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return "Unknown Dish"
        except Exception:
            return "Unknown Dish"

    def generate_recipe(self, prompt_text: str) -> tuple[str, float]:
        """
        Generates recipe using the FULL instruction prompt.
        Returns (generated_text, time_taken)
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.6,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        end_time = time.time()
        
        # Decode ONLY the new tokens (the recipe)
        input_length = inputs.input_ids.shape[1]
        generated_tokens = output[0][input_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response_text, end_time - start_time

    def run_comparison(self, dataset_sample, num_samples=None):
        results = []
        setup_gemini()

        # 1. Determine which rows to process
        # If num_samples is None or equals the dataset size, we use EVERYTHING provided.
        # This allows us to pass a fixed "Golden Set" of 20 recipes.
        if num_samples is not None and len(dataset_sample) > num_samples:
            print(f"🎲 Randomly sampling {num_samples} recipes from {len(dataset_sample)} options...")
            sample_indices = random.sample(range(len(dataset_sample)), num_samples)
            test_rows = [dataset_sample[i] for i in sample_indices]
        else:
            print(f"📏 Using fixed set of {len(dataset_sample)} recipes.")
            test_rows = dataset_sample

        for i, row in enumerate(test_rows):
            full_prompt = row['instruction']
            dish_name = self.extract_dish_name(full_prompt)
            
            print(f"\n[{i+1}/{len(test_rows)}] 🍛 DISH: {dish_name}")
            
            # --- 1. BASE MODEL ---
            if self.has_adapter:
                with self.model.disable_adapter():
                    recipe_text, gen_time = self.generate_recipe(full_prompt)
                    judge = judge_recipe(dish_name, recipe_text, "Base Model")
                    
                    results.append(BenchmarkResult(
                        model_name="Base Model (Untuned)",
                        dish_name=dish_name,
                        generation_time=round(gen_time, 2),
                        score=judge.get('score', 0),
                        reasoning=judge.get('reasoning', 'Error'),
                        recipe_length=len(recipe_text),
                        judge_model=judge.get('judge_model', 'Unknown')
                    ))

            # --- 2. FINE-TUNED MODEL ---
            model_label = "LoRA Fine-Tuned" if self.has_adapter else "Base Model Only"
            
            recipe_text, gen_time = self.generate_recipe(full_prompt)
            judge = judge_recipe(dish_name, recipe_text, model_label)

            results.append(BenchmarkResult(
                model_name=model_label,
                dish_name=dish_name,
                generation_time=round(gen_time, 2),
                score=judge.get('score', 0),
                reasoning=judge.get('reasoning', 'Error'),
                recipe_length=len(recipe_text),
                judge_model=judge.get('judge_model', 'Unknown')
            ))

        return results

def print_report(results: List[BenchmarkResult]):
    df = pd.DataFrame([vars(r) for r in results])
    
    print("\n" + "="*60)
    print("🏆 BENCHMARK FINAL REPORT")
    print("="*60)
    
    if not df.empty:
        print("\n📊 Average Scores:")
        # Group by the full model name to compare Configs
        print(df.groupby("model_name")[["score", "generation_time"]].mean())

        print("\n📝 Detailed Breakdown:")
        pd.set_option('display.max_colwidth', 50)
        print(df[["model_name", "judge_model", "dish_name", "score", "reasoning"]])
        
        output_file = "benchmark_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to '{output_file}'")
    else:
        print("No results generated.")

def run_llm_benchmark(test_dataset, model_configs: List[Dict]):
    all_results = []
    
    # We assume 'test_dataset' is ALREADY the fixed set of 20 recipes
    num_recipes = len(test_dataset)
    print(f"🏋️ Starting Fair Benchmark: {len(model_configs)} models x {num_recipes} same recipes")

    for config in model_configs:
        model_name = config["name"]
        base_id = config["base"]
        adapter = config.get("adapter", None)
        
        print(f"\n" + "="*60)
        print(f"🤖 CONFIGURATION: {model_name}")
        print("="*60)

        # 1. Clean Memory
        cleanup_resources()

        try:
            bench = RecipeBenchmark(base_model_id=base_id, adapter_path=adapter)
            
            # IMPORTANT: Pass num_samples=None so it consumes the whole fixed set
            model_results = bench.run_comparison(test_dataset, num_samples=None)
            
            for res in model_results:
                res.model_name = f"{model_name} - {res.model_name}"
                all_results.append(res)

            # 5. Explicit Deletion
            del bench.model
            del bench.base_model
            del bench.tokenizer
            del bench
            
        except Exception as e:
            print(f"❌ CRITICAL ERROR testing {model_name}: {e}")
        
        # 6. Clean Memory Again
        cleanup_resources()

    print_report(all_results)
