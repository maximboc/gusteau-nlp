"""
Constrained Recipe Generation using Outlines Framework.

This module uses Outlines to enforce structured recipe generation.
We use a REGEX constraint to force the model to follow the exact format it was trained on:
Ingredients: ...
Instructions: ...
This prevents "context drift" (hallucinations about stories, etc.).
"""

import time
import torch
import os
import re
from typing import List, Dict, Optional, Tuple
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.evaluation.judge_llm.judge_llm import cleanup_resources

try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError:
    OUTLINES_AVAILABLE = False
    print("⚠️ WARNING: Outlines not installed. Install with: pip install outlines")

from src.utils.utils import get_device

class ConstrainedRecipeBenchmark:
    """
    Uses Outlines for constrained generation to ensure valid recipe structure.
    """
    
    def __init__(self, base_model_id: str, adapter_path: Optional[str] = None, adapter_type: str = "unknown"):
        self.device_kwargs, _ = get_device()
        self.adapter_type = adapter_type
        print(f"🏗️ Initializing Constrained Benchmark with Base: {base_model_id}")
        
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
        self.model = self.base_model

        if adapter_path:
            if os.path.exists(adapter_path):
                print(f"🔗 Loading Adapter from: {adapter_path}")
                try:
                    self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
                    self.has_adapter = True
                    # Merge adapter for Outlines (it usually works better with merged weights or huggingface models)
                    # For now we pass the peft model directly, Outlines supports transformers models.
                except Exception as e:
                    print(f"❌ Error loading adapter (Fallback to Base): {e}")
            else:
                print(f"⚠️ WARNING: Adapter path '{adapter_path}' not found.")
                print("   -> Proceeding with Base Model only.")

        # 3. Initialize Outlines if available
        self.outlines_model = None
        if OUTLINES_AVAILABLE:
            try:
                # Outlines wraps the HF model - new API
                self.outlines_model = outlines.Transformers(
                    self.model,
                    self.tokenizer,
                )
                print("✅ Outlines initialized for constrained generation")
            except Exception as e:
                print(f"⚠️ Outlines initialization failed: {e}")
                print("   → Will fall back to standard generation")

    def generate_recipe_constrained(self, prompt_text: str) -> Tuple[str, float]:
        """
        Generates recipe using Outlines constrained generation.
        Enforces structure: Ingredients: ... Instructions: ...
        """
        # Ensure regex format matches training prompt structure
        # Training Format: "Instruction: {instruction}\n\nRecipe:"
        # 'prompt_text' passed here is usually just the instruction text "Create a detailed recipe for X".
        formatted_prompt = f"Instruction: {prompt_text}\n\nRecipe:"
        
        if not OUTLINES_AVAILABLE or self.outlines_model is None:
            return self.generate_recipe_standard(formatted_prompt)

        try:
            start_time = time.time()
            
            # --- The Constraint ---
            # Outlines regex pattern to enforce stricter structure:
            # 1. "Ingredients:" followed by non-empty content (at least one character)
            # 2. Two newlines
            # 3. "Instructions:" followed by non-empty content
            recipe_regex = r"Ingredients:[\s\S]+?\n\nInstructions:[\s\S]+"
            # Note: The +? is non-greedy, so it tries to find the *first* occurrence of \n\nInstructions:
            # This ensures we don't skip the Instructions header.
            
            # Using new outlines API: Generator(model, output_type)
            constraint = outlines.regex(recipe_regex)
            generator = outlines.Generator(self.outlines_model, constraint)
            
            # Outlines usually takes the prompt and continues it
            # The Generator.__call__ passes kwargs to the underlying sampler or model. 
            # Transformers model.generate uses 'max_new_tokens' usually.
            # Samplers are deprecated in newer outlines, we pass temperature directly.
            
            output = generator(formatted_prompt, max_new_tokens=600, temperature=0.6)
            
            # Output is already a string (the completion)
            output_text = output
            
            generation_time = time.time() - start_time
            return output_text, generation_time
            
        except Exception as e:
            print(f"   ❌ Constrained generation error: {e}")
            return self.generate_recipe_standard(formatted_prompt)

    def generate_recipe_standard(self, prompt_text: str) -> Tuple[str, float]:
        """
        Standard generation without constraints (fallback).
        """
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.6,
                do_sample=True,
                top_p=0.95,
            )
        
        generation_time = time.time() - start_time
        recipe_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the prompt from the output
        if recipe_text.startswith(prompt_text):
            recipe_text = recipe_text[len(prompt_text):].strip()
            
        return recipe_text, generation_time

    def generate_recipe(self, prompt_text: str) -> Tuple[str, float]:
        """Main generation method - uses constrained if available, falls back to standard."""
        return self.generate_recipe_constrained(prompt_text)


def run_constrained_benchmark(test_dataset, model_configs):
    """
    Runs benchmark with constrained generation using Outlines.
    """
    print(f"\n🎯 Starting Outlines Constrained Benchmark on {len(test_dataset)} samples...")
    
    results = {}

    for config in model_configs:
        model_name = config["name"]
        base_id = config["base"]
        adapter = config.get("adapter", None)
        
        print(f"\n🔹 Evaluating Model with Outlines: {model_name}")
        
        # Extract adapter type from model name
        adapter_type = "unknown"
        if "(" in model_name and ")" in model_name:
            adapter_type = model_name.split("(")[1].split(")")[0]
        
        # Create benchmark with constrained generation
        # NOTE: We reload the model here. Ideally we would pass the loaded model, 
        # but the class structure keeps things isolated.
        benchmark = ConstrainedRecipeBenchmark(base_id, adapter, adapter_type=adapter_type)
        
        total_time = 0
        recipes_generated = []
        
        print(f"   Generating constrained recipes...")
        
        for i, row in enumerate(test_dataset):
            prompt = row["instruction"]
            
            # Generate with constraints
            generated_text, gen_time = benchmark.generate_recipe(prompt)
            
            total_time += gen_time
            recipes_generated.append(generated_text)
            
            # Check if output contains both required sections
            # The regex *should* guarantee this, but let's verify
            has_ingredients = "Ingredients" in generated_text
            has_instructions = "Instructions" in generated_text
            structure_valid = "✓" if (has_ingredients and has_instructions) else "✗"
            
            print(f"   [{i+1}/{len(test_dataset)}] Time: {gen_time:.2f}s | Structure: {structure_valid}")
            # print(f"      Preview: {generated_text[:50]}...")

        avg_time = total_time / len(test_dataset)
        valid_count = sum(1 for r in recipes_generated if "Ingredients" in r and "Instructions" in r)
        structure_validity = (valid_count / len(recipes_generated)) * 100
        
        results[model_name] = {
            "avg_generation_time": avg_time,
            "structure_validity": structure_validity,
            "valid_recipes": valid_count,
            "total_recipes": len(recipes_generated)
        }
        
        # Cleanup memory
        del benchmark
        cleanup_resources()

    print("\n🎯 Outlines Results Summary 🎯")
    print("-" * 70)
    print(f"{'Model':<30} | {'Valid Structure %':<15} | {'Avg Time (s)':<12}")
    print("-" * 70)
    for model, metrics in results.items():
        print(f"{model:<30} | {metrics['structure_validity']:>6.1f}%            | {metrics['avg_generation_time']:>6.2f}s")
    print("-" * 70)
