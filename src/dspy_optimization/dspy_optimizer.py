import dspy
from dspy.teleprompt import BootstrapFewShot
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import os

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
    model_output = pred.recipe
    
    score = 0
    if "Ingredients:" in model_output or "**Ingredients**" in model_output:
        score += 1
    if "Instructions:" in model_output or "**Instructions**" in model_output:
        score += 1
    if len(model_output) > 200: # Ensure it's not too short
        score += 1
        
    return score >= 3 # Pass if all conditions met

def check_dspy_optimization(model_name, dataset_dict, adapter_path=None, output_path="models/dspy_compiled_program.json"):
    """
    Runs DSPy optimization to create an optimized prompt program.
    
    NOTE: DSPy (via LiteLLM) requires API-based models or inference servers.
    For local HuggingFace models loaded directly with transformers, DSPy cannot be used.
    
    Options to use DSPy with local models:
    1. Deploy model to vLLM server
    2. Deploy model to Text Generation Inference (TGI) server  
    3. Use Ollama with a downloaded model
    4. Use OpenAI-compatible API wrapper
    
    For this project using local PEFT adapters, DSPy integration is not straightforward.
    """
    print(f"⚠️ DSPy optimization skipped for local model: {model_name}")
    print(f"   DSPy requires API-based models or hosted inference endpoints.")
    print(f"   Your local HuggingFace model with PEFT adapter cannot be used directly.")
    print(f"   To use DSPy, consider deploying to vLLM, TGI, or Ollama.")
    return None, None
    
    # Create the training set for DSPy (needs dspy.Example)
    # We take a small subset for 'few-shot' optimization
    print("Preparing DSPy dataset...")
    train_subset = dataset_dict['train'].select(range(20)) # 20 examples for bootstrap
    
    trainset = []
    for item in train_subset:
        # Our dataset has 'instruction' (input) and 'output' (recipe)
        # We need to extract the dish name from the instruction if possible, or just use the instruction as input.
        # The prompt is "You are a chef... User: {input}"
        # Let's assume input is "Make me a [Dish Name]"
        
        # We map dataset 'instruction' -> 'dish_name' and 'output' -> 'recipe'
        trainset.append(dspy.Example(dish_name=item['instruction'], recipe=item['output']).with_inputs('dish_name'))
        
    print("Compiling DSPy program with BootstrapFewShot...")
    
    # Define the optimizer
    optimizer = BootstrapFewShot(metric=validate_recipe_structure, max_bootstrapped_demos=4, max_labeled_demos=4)
    
    # Compile
    recipe_program = RecipeModule()
    compiled_recipe_program = optimizer.compile(recipe_program, trainset=trainset)
    
    print("Optimization complete.")
    
    # Save the compiled program
    compiled_recipe_program.save(output_path)
    print(f"Compiled program saved to {output_path}")
    
    return compiled_recipe_program, lm

def run_dspy_benchmark(test_dataset, compiled_program):
    """
    Runs generation using the compiled DSPy program.
    """
    print("\n--- Running DSPy Benchmark ---")
    results = []
    
    for item in test_dataset:
        dish_name = item['instruction']
        print(f"Generating for: {dish_name}")
        
        try:
            pred = compiled_program(dish_name=dish_name)
            generated_recipe = pred.recipe
            results.append({
                "dish": dish_name,
                "generated_recipe": generated_recipe
            })
            print(f"Generated length: {len(generated_recipe)}")
        except Exception as e:
            print(f"Error generating with DSPy: {e}")
            
    return results
