import dspy
from dspy.teleprompt import BootstrapFewShot
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os
import warnings

class RecipeSignature(dspy.Signature):
    """
    Generates a detailed cooking recipe based on the dish name and cuisine style.
    The output should follow a structured format with Ingredients and Instructions.
    """
    dish_name = dspy.InputField(desc="The name of the dish to cook")
    recipe = dspy.OutputField(desc="A detailed recipe including Ingredients and Instructions")

class RecipeModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_recipe = dspy.ChainOfThought(RecipeSignature)
    
    def forward(self, dish_name):
        prediction = self.generate_recipe(dish_name=dish_name)
        return dspy.Prediction(recipe=prediction.recipe)

def validate_recipe_structure(example, pred, trace=None):
    """
    A simple metric to check if the generated recipe has the basic structure.
    Checks for 'Ingredients' and 'Instructions' keywords.
    """
    try:
        model_output = pred.recipe if hasattr(pred, 'recipe') else str(pred)
        
        score = 0
        if "Ingredients:" in model_output or "**Ingredients**" in model_output or "ingredients" in model_output.lower():
            score += 1
        if "Instructions:" in model_output or "**Instructions**" in model_output or "instructions" in model_output.lower():
            score += 1
        if len(model_output) > 200: # Ensure it's not too short
            score += 1
            
        return score >= 2 # Pass if at least 2 conditions met
    except Exception as e:
        print(f"Validation error: {e}")
        return False

class CustomLocalLM:
    """
    Custom wrapper for local HuggingFace models to work with DSPy.
    This bypasses the need for API endpoints.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def __call__(self, prompt=None, messages=None, **kwargs):
        """Generate text using the local model."""
        if messages:
            # Convert messages to prompt
            prompt_text = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                prompt_text += f"{role}: {content}\n"
            prompt_text += "assistant:"
        else:
            prompt_text = prompt or ""
        
        # Generate
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        max_new_tokens = kwargs.get('max_tokens', 512)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=kwargs.get('temperature', 0.7),
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the output
        if prompt_text in generated_text:
            generated_text = generated_text[len(prompt_text):].strip()
        
        # Return in format compatible with DSPy
        return [generated_text]

def check_dspy_optimization(model_name, dataset_dict, adapter_path=None, output_path="models/dspy_compiled_program.json"):
    """
    Creates a DSPy program using local HuggingFace model with PEFT adapter.
    NOTE: Full DSPy optimization (BootstrapFewShot) requires API-based models.
    This creates a basic DSPy program without optimization for local models.
    """
    print(f"🔧 Initializing DSPy with local model: {model_name}")
    print("⚠️  Note: Using local model without BootstrapFewShot optimization")
    print("   (Full DSPy optimization requires API-based models)")
    
    try:
        # Load the local model with adapter
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load base model with 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # Load adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            print(f"🔗 Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path)
        else:
            model = base_model
        
        # Create custom LM wrapper
        custom_lm = CustomLocalLM(model, tokenizer)
        
        # Create a simple recipe program (without optimization)
        recipe_program = RecipeModule()
        
        print("✅ DSPy program created (using local model without optimization)")
        
        return recipe_program, custom_lm
        
    except Exception as e:
        print(f"❌ DSPy initialization failed: {e}")
        print(f"   Local model integration encountered an error.")
        import traceback
        traceback.print_exc()
        return None, None

def run_dspy_benchmark(test_dataset, compiled_program, custom_lm=None):
    """
    Runs generation using the DSPy program with custom local LM.
    """
    print("\n🎯 Running DSPy Benchmark")
    print(f"Generating recipes for {len(test_dataset)} test samples...")
    print("-" * 60)
    
    results = []
    
    for idx, item in enumerate(test_dataset, 1):
        dish_name = item['instruction']
        expected_recipe = item['output']
        
        try:
            print(f"   [{idx}/{len(test_dataset)}] Generating: {dish_name[:50]}...")
            
            # Use custom LM directly if provided
            if custom_lm:
                prompt = f"""You are a professional chef. Generate a detailed recipe for the following dish.

Dish: {dish_name}

Recipe (with Ingredients and Instructions):"""
                
                generated_text = custom_lm(prompt=prompt)[0]
                pred_recipe = generated_text
            else:
                # Use DSPy program (if using API models)
                pred = compiled_program(dish_name=dish_name)
                pred_recipe = pred.recipe
            
            # Validate structure
            has_structure = "ingredient" in pred_recipe.lower() and "instruction" in pred_recipe.lower()
            structure_mark = "✓" if has_structure else "✗"
            
            results.append({
                "dish": dish_name,
                "generated_recipe": pred_recipe,
                "expected_recipe": expected_recipe,
                "has_structure": has_structure
            })
            
            print(f"               Structure: {structure_mark} | Length: {len(pred_recipe)} chars")
            
        except Exception as e:
            print(f"   [{idx}/{len(test_dataset)}] ❌ Error: {e}")
            results.append({
                "dish": dish_name,
                "generated_recipe": "",
                "expected_recipe": expected_recipe,
                "has_structure": False
            })
    
    # Summary statistics
    print("\n" + "="*60)
    print("🏆 DSPy Benchmark Results Summary")
    print("="*60)
    
    valid_structures = sum(1 for r in results if r['has_structure'])
    avg_length = sum(len(r['generated_recipe']) for r in results) / len(results) if results else 0
    
    print(f"Total Samples:       {len(results)}")
    print(f"Valid Structures:    {valid_structures} ({valid_structures/len(results)*100:.1f}%)")
    print(f"Avg Recipe Length:   {avg_length:.0f} characters")
    print("="*60)
    
    return results