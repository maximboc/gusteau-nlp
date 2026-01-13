import time
import nltk
from rapidfuzz import fuzz
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.evaluation.judge_llm.judge_llm import RecipeBenchmark, cleanup_resources
from src.evaluation.quantitative.recipe_metrics import RecipeMetricsEvaluator

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def calculate_metrics(reference_text, generated_text):
    """
    Computes quantitative metrics:
    1. BLEU Score (N-gram precision)
    2. Levenshtein Similarity (Fuzz Ratio)
    3. Recipe-Specific Metrics (Ingredient Coverage, Temperature Validation, Allergen Handling)
    """
    # Tokenize for BLEU
    ref_tokens = nltk.word_tokenize(reference_text.lower())
    gen_tokens = nltk.word_tokenize(generated_text.lower())
    
    # BLEU
    # sentence_bleu expects a list of reference token lists: [[ref1_tokens, ref2_tokens], ...]
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)
    
    # Levenshtein / Fuzz Ratio (0-100)
    similarity_score = fuzz.ratio(reference_text.lower(), generated_text.lower())
    
    # Recipe-Specific Metrics
    recipe_evaluator = RecipeMetricsEvaluator()
    recipe_metrics = recipe_evaluator.evaluate(reference_text, generated_text)
    
    return {
        "bleu": bleu_score,
        "similarity": similarity_score,
        "recipe_metrics": recipe_metrics
    }

def run_quantitative_benchmark(test_dataset, model_configs):
    """
    Runs quantitative evaluation (BLEU & Similarity) on the test dataset.
    """
    print(f"\n📊 Starting Quantitative Benchmark on {len(test_dataset)} samples...")
    
    results = {}

    for config in model_configs:
        model_name = config["name"]
        print(f"\n🔹 Evaluating Model: {model_name}")
        
        # Reuse existing class for loading & generation
        benchmark = RecipeBenchmark(config["base"], config["adapter"])
        
        total_bleu = 0
        total_similarity = 0
        total_ingredient_coverage = 0
        total_temperature_validation = 0
        total_allergen_handling = 0
        total_composite_score = 0
        
        print(f"   Generating recipes...")
        
        for i, row in enumerate(test_dataset):
            prompt = row["instruction"]
            reference = row["output"]
            
            # Generate
            generated_text, _ = benchmark.generate_recipe(prompt)
            
            # Post-processing to remove the instruction part if the model repeats it
            # (Simple heuristic: if generated text starts with instruction, strip it. 
            # But the model usually continues. We'll evaluate the full generated string.)
            
            # Calculate Metrics
            metrics = calculate_metrics(reference, generated_text)
            
            total_bleu += metrics["bleu"]
            total_similarity += metrics["similarity"]
            total_ingredient_coverage += metrics["recipe_metrics"]["ingredient_coverage"]["score"]
            total_temperature_validation += metrics["recipe_metrics"]["temperature_validation"]["score"]
            total_allergen_handling += metrics["recipe_metrics"]["allergen_handling"]["score"]
            total_composite_score += metrics["recipe_metrics"]["composite_score"]
            
            print(f"   [{i+1}/{len(test_dataset)}] BLEU: {metrics['bleu']:.4f} | Sim: {metrics['similarity']:.1f} | Recipe: {metrics['recipe_metrics']['composite_score']:.4f}")

        # Averages
        avg_bleu = total_bleu / len(test_dataset)
        avg_sim = total_similarity / len(test_dataset)
        avg_ingredient_coverage = total_ingredient_coverage / len(test_dataset)
        avg_temperature_validation = total_temperature_validation / len(test_dataset)
        avg_allergen_handling = total_allergen_handling / len(test_dataset)
        avg_composite_score = total_composite_score / len(test_dataset)
        
        results[model_name] = {
            "avg_bleu": avg_bleu,
            "avg_similarity": avg_sim,
            "avg_ingredient_coverage": avg_ingredient_coverage,
            "avg_temperature_validation": avg_temperature_validation,
            "avg_allergen_handling": avg_allergen_handling,
            "avg_composite_score": avg_composite_score
        }
        
        # Cleanup memory
        del benchmark
        cleanup_resources()

    print("\n🏆 Quantitative Results Summary 🏆")
    print("-" * 120)
    print(f"{'Model':<25} | {'BLEU':<10} | {'Similarity':<12} | {'Ingredients':<12} | {'Temperature':<12} | {'Allergen':<12} | {'Composite':<10}")
    print("-" * 120)
    for model, metrics in results.items():
        print(f"{model:<25} | {metrics['avg_bleu']:.4f}     | {metrics['avg_similarity']:.1f}       | {metrics['avg_ingredient_coverage']:.4f}      | {metrics['avg_temperature_validation']:.4f}       | {metrics['avg_allergen_handling']:.4f}     | {metrics['avg_composite_score']:.4f}")
    print("-" * 120)
