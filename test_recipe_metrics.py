"""
Test script for recipe-specific domain metrics.
Demonstrates the functionality of ingredient coverage, temperature validation, and allergen handling.
"""

from src.evaluation.quantitative.recipe_metrics import RecipeMetricsEvaluator

def test_ingredient_coverage():
    """Test ingredient coverage metric"""
    print("\n" + "="*70)
    print("TEST 1: INGREDIENT COVERAGE METRIC")
    print("="*70)
    
    reference = """
    Ingredients: flour, eggs, butter, sugar, salt
    
    Instructions:
    1. Mix butter and sugar
    2. Add eggs one by one
    3. Fold in flour and salt
    """
    
    # Good recipe - uses all ingredients
    generated_good = """
    Mix the flour with salt in a bowl.
    In another bowl, cream together butter and sugar.
    Beat in eggs one at a time until well combined.
    Fold the flour mixture into the egg mixture.
    Bake at 180°C for 25 minutes.
    """
    
    # Poor recipe - missing some ingredients
    generated_poor = """
    Add flour to a bowl.
    Mix with sugar.
    Bake at 180°C.
    """
    
    evaluator = RecipeMetricsEvaluator()
    result_good = evaluator.evaluate(reference, generated_good)
    result_poor = evaluator.evaluate(reference, generated_poor)
    
    print(f"\n✓ Good Recipe (uses all ingredients):")
    print(f"  Ingredient Coverage Score: {result_good['ingredient_coverage']['score']:.3f}")
    print(f"  Coverage: {result_good['ingredient_coverage']['used_count']}/{result_good['ingredient_coverage']['total_count']} ({result_good['ingredient_coverage']['coverage_percentage']:.1f}%)")
    
    print(f"\n✗ Poor Recipe (missing ingredients):")
    print(f"  Ingredient Coverage Score: {result_poor['ingredient_coverage']['score']:.3f}")
    print(f"  Coverage: {result_poor['ingredient_coverage']['used_count']}/{result_poor['ingredient_coverage']['total_count']} ({result_poor['ingredient_coverage']['coverage_percentage']:.1f}%)")
    print(f"  Missing: {result_poor['ingredient_coverage']['missing_ingredients']}")


def test_temperature_validation():
    """Test temperature validation metric"""
    print("\n" + "="*70)
    print("TEST 2: TEMPERATURE VALIDATION METRIC")
    print("="*70)
    
    reference = "Bake at 180°C for 25 minutes"
    
    # Good - realistic temperatures
    generated_good = """
    Preheat oven to 180°C.
    Bake for 25 minutes until golden.
    The internal temperature should reach 65°C.
    """
    
    # Bad - unrealistic temperatures
    generated_bad = """
    Preheat oven to 500°C.
    Boil water at 200°C.
    Fry at 350°C.
    """
    
    evaluator = RecipeMetricsEvaluator()
    result_good = evaluator.evaluate(reference, generated_good)
    result_bad = evaluator.evaluate(reference, generated_bad)
    
    print(f"\n✓ Good Recipe (realistic temps):")
    print(f"  Temperature Validation Score: {result_good['temperature_validation']['score']:.3f}")
    print(f"  Temperatures Found: {result_good['temperature_validation']['valid_count']}")
    print(f"  All Valid: {result_good['temperature_validation']['all_valid']}")
    
    print(f"\n✗ Bad Recipe (unrealistic temps):")
    print(f"  Temperature Validation Score: {result_bad['temperature_validation']['score']:.3f}")
    print(f"  Valid Count: {result_bad['temperature_validation']['valid_count']}")
    print(f"  Invalid Count: {result_bad['temperature_validation']['invalid_count']}")


def test_allergen_handling():
    """Test allergen handling metric"""
    print("\n" + "="*70)
    print("TEST 3: ALLERGEN HANDLING METRIC")
    print("="*70)
    
    reference = "Ingredients: peanut butter, eggs, milk, wheat flour"
    
    # Good - mentions allergens and offers alternatives
    generated_good = """
    Mix peanut butter with eggs (this recipe contains nuts - allergy warning).
    You can substitute with sunflower butter if allergic to peanuts.
    Use dairy-free milk as an alternative for dairy-free option.
    This contains gluten - use gluten-free flour for celiac-safe version.
    """
    
    # Poor - no allergen awareness
    generated_poor = """
    Mix peanut butter with eggs.
    Add milk and flour.
    Stir well and bake.
    """
    
    evaluator = RecipeMetricsEvaluator()
    result_good = evaluator.evaluate(reference, generated_good)
    result_poor = evaluator.evaluate(reference, generated_poor)
    
    print(f"\n✓ Good Recipe (allergen-aware):")
    print(f"  Allergen Handling Score: {result_good['allergen_handling']['score']:.3f}")
    print(f"  Allergens Mentioned: {result_good['allergen_handling']['allergens_mentioned']}")
    print(f"  Substitution Mentions: {result_good['allergen_handling']['substitution_count']}")
    print(f"  Safety Warnings: {result_good['allergen_handling']['has_safety_warnings']}")
    
    print(f"\n✗ Poor Recipe (no allergen awareness):")
    print(f"  Allergen Handling Score: {result_poor['allergen_handling']['score']:.3f}")
    print(f"  Allergens Mentioned: {result_poor['allergen_handling']['allergens_mentioned']}")
    print(f"  Substitution Mentions: {result_poor['allergen_handling']['substitution_count']}")
    print(f"  Safety Warnings: {result_poor['allergen_handling']['has_safety_warnings']}")


def test_composite_score():
    """Test composite recipe score"""
    print("\n" + "="*70)
    print("TEST 4: COMPOSITE RECIPE SCORE (All Metrics Combined)")
    print("="*70)
    
    reference = """
    Ingredients: butter, sugar, eggs, flour, salt, milk
    
    Instructions:
    Cream butter and sugar. Add eggs. Mix flour and salt.
    Combine with milk. Bake at 180°C for 30 minutes.
    """
    
    # Excellent recipe
    generated_excellent = """
    Cream together 100g butter and 150g sugar until light and fluffy.
    Add 3 eggs one at a time, beating well after each addition.
    Combine 200g flour with 1/2 tsp salt in a separate bowl.
    Fold the dry ingredients into the wet mixture alternately with 100ml milk.
    
    Note: This recipe contains eggs, dairy, and wheat - not suitable for those with allergies.
    You can use dairy-free milk as a substitute.
    
    Pour into a lined pan and bake at 180°C for 30-35 minutes until a skewer comes out clean.
    """
    
    evaluator = RecipeMetricsEvaluator()
    result = evaluator.evaluate(reference, generated_excellent)
    
    print(f"\n📊 COMPREHENSIVE RECIPE EVALUATION:")
    print(f"\n  Ingredient Coverage:        {result['ingredient_coverage']['score']:.3f}")
    print(f"    └─ Used: {result['ingredient_coverage']['used_count']}/{result['ingredient_coverage']['total_count']} ({result['ingredient_coverage']['coverage_percentage']:.1f}%)")
    
    print(f"\n  Temperature Validation:     {result['temperature_validation']['score']:.3f}")
    print(f"    └─ Temperatures Found: {result['temperature_validation']['valid_count']}")
    print(f"    └─ All Valid: {result['temperature_validation']['all_valid']}")
    
    print(f"\n  Allergen Handling:          {result['allergen_handling']['score']:.3f}")
    print(f"    └─ Allergens Mentioned: {len(result['allergen_handling']['allergens_mentioned'])}")
    print(f"    └─ Substitutions: {result['allergen_handling']['substitution_count']}")
    print(f"    └─ Safety Warnings: {result['allergen_handling']['has_safety_warnings']}")
    
    print(f"\n  ⭐ COMPOSITE RECIPE SCORE:   {result['composite_score']:.3f}/1.0")
    print(f"     (Average of all three metrics)")


if __name__ == "__main__":
    print("\n🧪 RECIPE METRICS VALIDATION TEST SUITE\n")
    
    test_ingredient_coverage()
    test_temperature_validation()
    test_allergen_handling()
    test_composite_score()
    
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETED")
    print("="*70 + "\n")
