"""
Recipe-Specific Domain Metrics for Evaluation

This module implements domain-aware metrics tailored to recipe generation:
1. Ingredient Coverage - verifies if all listed ingredients are used in instructions
2. Temperature Validation - checks if cooking temperatures are realistic
3. Allergen Handling - detects and scores allergen awareness
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Common culinary allergens
COMMON_ALLERGENS = {
    'peanut', 'peanuts', 'tree nut', 'almond', 'walnut', 'cashew',
    'milk', 'dairy', 'cheese', 'butter', 'cream', 'yogurt',
    'egg', 'eggs', 'shellfish', 'shrimp', 'crab', 'lobster',
    'fish', 'salmon', 'tuna', 'cod',
    'soy', 'soybean', 'tofu',
    'wheat', 'gluten', 'barley', 'rye',
    'sesame', 'mustard'
}

# Temperature ranges in Celsius (culinary standards)
COOKING_TEMPS = {
    'oven': (100, 300),  # 100-300°C typical oven range
    'stovetop': (0, 200),  # Stovetop generally lower
    'boil': (100, 105),  # Water boils at 100°C
    'simmer': (80, 100),
    'fry': (150, 200),
    'bake': (150, 250),
    'roast': (160, 220),
}


@dataclass
class IngredientCoverageResult:
    """Result of ingredient coverage evaluation"""
    score: float  # 0-1, higher is better
    coverage_percentage: float  # % of ingredients mentioned
    listed_ingredients: List[str]
    used_ingredients: List[str]
    missing_ingredients: List[str]
    unused_ingredients: List[str]


@dataclass
class TemperatureValidationResult:
    """Result of temperature validation"""
    score: float  # 0-1, higher is better
    has_temperature_mentions: bool
    temperatures_found: List[Tuple[float, str]]  # (temp_value, unit)
    all_valid: bool
    invalid_temperatures: List[Tuple[float, str]]


@dataclass
class AllergenHandlingResult:
    """Result of allergen handling evaluation"""
    score: float  # 0-1, higher is better
    allergens_mentioned: List[str]
    allergen_awareness: bool  # True if allergens explicitly mentioned or substitutions offered
    substitution_mentions: int
    safety_notes_present: bool


class IngredientCoverageMetric:
    """
    Evaluates whether generated recipes use all listed ingredients.
    
    Metric: (ingredients_used / ingredients_listed)
    High score means recipe uses most/all listed ingredients.
    """
    
    @staticmethod
    def extract_ingredients(text: str) -> List[str]:
        """
        Extract ingredient names from text.
        Handles both:
        - Python list format: ['item1', 'item2'] (from reference)
        - Comma-separated: item1, item2 (from generated)
        """
        # Look for "Ingredients:" section
        ingredient_match = re.search(
            r'ingredients?:\s*(.+?)(?:instructions?:|steps?:|$)',
            text,
            re.IGNORECASE | re.DOTALL
        )
        
        if ingredient_match:
            ingredients_text = ingredient_match.group(1).strip()
            
            # Check if it's a Python list format (starts with [ or contains [')
            if ingredients_text.startswith('[') or "['" in ingredients_text:
                # Parse Python list string: ['item1', 'item2', ...]
                # Extract items between quotes
                ingredients = re.findall(r"'([^']+)'", ingredients_text)
            else:
                # Comma-separated format
                ingredients = re.split(r'[,\n]', ingredients_text)
            
            # Clean up: remove whitespace, quotes, brackets, lowercase
            ingredients = [
                ing.strip().lower().strip("'[]\"")
                for ing in ingredients
                if ing.strip() and ing.strip() not in ["[]", "['']", "''"]
            ]
            return ingredients
        
        return []
    
    @staticmethod
    def score_coverage(reference_ingredients: List[str], 
                       generated_text: str) -> IngredientCoverageResult:
        """
        Calculate ingredient coverage: what % of listed ingredients appear in instructions?
        """
        # Normalize reference ingredients and extract core terms
        ref_ingr = [ing.lower().strip() for ing in reference_ingredients]
        gen_text_lower = generated_text.lower()
        
        used_ingredients = []
        for ingredient in ref_ingr:
            # Extract core ingredient name (first significant word)
            # e.g., "fresh basil leaves" -> check for "basil"
            core_words = [w for w in ingredient.split() if len(w) > 2]
            
            # Check if full ingredient or core words appear in text
            found = False
            
            # First try exact ingredient match
            if re.search(r'\b' + re.escape(ingredient) + r'\b', gen_text_lower):
                found = True
            # Then try matching core words (avoid too-generic matches)
            elif core_words:
                for word in core_words:
                    # Skip common words
                    if word in ['cup', 'cups', 'tsp', 'tbsp', 'tablespoon', 'teaspoon', 'ounce', 'oz', 'gram', 'fresh', 'dried', 'ground']:
                        continue
                    if re.search(r'\b' + re.escape(word) + r'\b', gen_text_lower):
                        found = True
                        break
            
            if found:
                used_ingredients.append(ingredient)
        
        unused_ingredients = [ing for ing in ref_ingr if ing not in used_ingredients]
        
        # Calculate coverage percentage
        if len(ref_ingr) > 0:
            coverage_percentage = (len(used_ingredients) / len(ref_ingr)) * 100
            coverage_score = len(used_ingredients) / len(ref_ingr)
        else:
            coverage_percentage = 0.0
            coverage_score = 0.0
        
        return IngredientCoverageResult(
            score=coverage_score,
            coverage_percentage=coverage_percentage,
            listed_ingredients=ref_ingr,
            used_ingredients=used_ingredients,
            missing_ingredients=unused_ingredients,
            unused_ingredients=[]
        )


class TemperatureValidationMetric:
    """
    Validates cooking temperatures mentioned in recipes.
    
    Checks:
    - Are temperatures realistic for cooking methods?
    - Are they in reasonable ranges?
    - Celsius vs Fahrenheit interpretation
    """
    
    @staticmethod
    def extract_temperatures(text: str) -> List[Tuple[float, str]]:
        """
        Extract temperature values and units from text.
        Returns list of (value, unit) tuples.
        """
        temperatures = []
        
        # Pattern for: "X degrees C/F" or "X°C/F" or "X C/F" or "X celsius/fahrenheit"
        patterns = [
            r'(\d+\.?\d*)\s*(?:degrees?|°)\s*([CF])',  # "180 degrees C" or "180°F"
            r'(\d+\.?\d*)\s*([CF])\b',  # "180C" or "350F"
            r'(\d+\.?\d*)\s*(?:degrees?)?\s*(celsius|fahrenheit)',  # "180 celsius" or "350 degrees fahrenheit"
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    temp_value = float(match.group(1))
                    unit = match.group(2).upper()
                    # Normalize 'celsius' -> 'C', 'fahrenheit' -> 'F'
                    if unit.startswith('C'):
                        unit = 'C'
                    elif unit.startswith('F'):
                        unit = 'F'
                    temperatures.append((temp_value, unit))
                except (ValueError, IndexError):
                    pass
        
        return temperatures
    
    @staticmethod
    def is_temperature_valid(temp_value: float, unit: str) -> bool:
        """
        Check if a temperature is realistic for cooking.
        """
        # Normalize to Celsius
        if unit.upper() == 'F':
            temp_celsius = (temp_value - 32) * 5 / 9
        else:
            temp_celsius = temp_value
        
        # Reasonable cooking temperatures: -20°C to 250°C
        # (includes freezing, oven baking, but not extreme heat)
        return -20 <= temp_celsius <= 250
    
    @staticmethod
    def score_temperatures(generated_text: str) -> TemperatureValidationResult:
        """
        Evaluate temperature mentions for validity.
        """
        temps = TemperatureValidationMetric.extract_temperatures(generated_text)
        
        valid_temps = []
        invalid_temps = []
        
        for temp_val, unit in temps:
            if TemperatureValidationMetric.is_temperature_valid(temp_val, unit):
                valid_temps.append((temp_val, unit))
            else:
                invalid_temps.append((temp_val, unit))
        
        has_temps = len(temps) > 0
        
        # Score: if no temperatures mentioned, neutral (0.5)
        # If temperatures present, score based on validity
        if has_temps:
            temperature_score = len(valid_temps) / len(temps)
        else:
            temperature_score = 0.5  # Neutral if no temps mentioned
        
        return TemperatureValidationResult(
            score=temperature_score,
            has_temperature_mentions=has_temps,
            temperatures_found=valid_temps,
            all_valid=len(invalid_temps) == 0,
            invalid_temperatures=invalid_temps
        )


class AllergenHandlingMetric:
    """
    Evaluates whether the recipe acknowledges common allergens.
    
    Checks:
    - Are common allergens mentioned?
    - Are substitutions offered?
    - Are safety warnings present?
    """
    
    @staticmethod
    def extract_allergens(text: str) -> List[str]:
        """
        Find mentions of common allergens in the text.
        """
        text_lower = text.lower()
        found_allergens = []
        
        for allergen in COMMON_ALLERGENS:
            if re.search(r'\b' + re.escape(allergen) + r'\b', text_lower):
                found_allergens.append(allergen)
        
        return found_allergens
    
    @staticmethod
    def count_substitution_mentions(text: str) -> int:
        """
        Count mentions of substitutions or alternatives.
        """
        substitution_keywords = [
            r'\bsubstitut',  # substitute, substitution
            r'\balt(?:ernativ)?',  # alternative, alt
            r'\bif you.*(?:can\'t|don\'t have|allergic)',  # conditional alternatives
            r'\binstead of\b',
            r'\bor use\b',
            r'\byou can use\b',
        ]
        
        count = 0
        text_lower = text.lower()
        for pattern in substitution_keywords:
            count += len(re.findall(pattern, text_lower))
        
        return count
    
    @staticmethod
    def has_safety_warnings(text: str) -> bool:
        """
        Check for safety-related keywords.
        """
        safety_keywords = [
            r'warning',
            r'caution',
            r'allergic',
            r'allergen',
            r'gluten.free',
            r'dairy.free',
            r'nut.free',
            r'shellfish',
            r'contains',
        ]
        
        text_lower = text.lower()
        for keyword in safety_keywords:
            if re.search(r'\b' + keyword, text_lower):
                return True
        
        return False
    
    @staticmethod
    def score_allergen_handling(generated_text: str, 
                               reference_ingredients: List[str]) -> AllergenHandlingResult:
        """
        Evaluate allergen awareness in the recipe.
        """
        allergens = AllergenHandlingMetric.extract_allergens(generated_text)
        substitutions = AllergenHandlingMetric.count_substitution_mentions(generated_text)
        has_warnings = AllergenHandlingMetric.has_safety_warnings(generated_text)
        
        # Allergen awareness: recipe mentions allergens or offers substitutions
        allergen_awareness = (len(allergens) > 0) or (substitutions > 0) or has_warnings
        
        # Scoring logic:
        # - Base: 0.3 for any allergen mention
        # - +0.3 for substitution mentions
        # - +0.4 for explicit safety warnings
        score = 0.0
        if len(allergens) > 0:
            score += 0.3
        if substitutions > 0:
            score += 0.3
        if has_warnings:
            score += 0.4
        
        # Cap at 1.0
        score = min(score, 1.0)
        
        return AllergenHandlingResult(
            score=score,
            allergens_mentioned=allergens,
            allergen_awareness=allergen_awareness,
            substitution_mentions=substitutions,
            safety_notes_present=has_warnings
        )


class RecipeMetricsEvaluator:
    """
    Unified interface for all recipe-specific metrics.
    """
    
    def __init__(self):
        self.ingredient_metric = IngredientCoverageMetric()
        self.temperature_metric = TemperatureValidationMetric()
        self.allergen_metric = AllergenHandlingMetric()
    
    def evaluate(self, reference_text: str, generated_text: str) -> Dict:
        """
        Evaluate all recipe metrics.
        
        Args:
            reference_text: Original recipe with listed ingredients
            generated_text: Generated recipe instructions
        
        Returns:
            Dictionary with all metric results
        """
        # Extract reference ingredients
        ref_ingredients = IngredientCoverageMetric.extract_ingredients(reference_text)
        
        # Score each metric
        ingredient_result = self.ingredient_metric.score_coverage(ref_ingredients, generated_text)
        temperature_result = self.temperature_metric.score_temperatures(generated_text)
        allergen_result = self.allergen_metric.score_allergen_handling(generated_text, ref_ingredients)
        
        # Composite score (weighted by importance)
        # Ingredient coverage is most critical (40%), temperature safety (40%), allergen awareness (20%)
        composite_score = (
            ingredient_result.score * 0.40 +
            temperature_result.score * 0.40 +
            allergen_result.score * 0.20
        )
        
        return {
            'composite_score': composite_score,
            'ingredient_coverage': {
                'score': ingredient_result.score,
                'coverage_percentage': ingredient_result.coverage_percentage,
                'used_count': len(ingredient_result.used_ingredients),
                'total_count': len(ingredient_result.listed_ingredients),
                'missing_ingredients': ingredient_result.missing_ingredients,
            },
            'temperature_validation': {
                'score': temperature_result.score,
                'has_temperatures': temperature_result.has_temperature_mentions,
                'valid_count': len(temperature_result.temperatures_found),
                'invalid_count': len(temperature_result.invalid_temperatures),
                'all_valid': temperature_result.all_valid,
            },
            'allergen_handling': {
                'score': allergen_result.score,
                'allergens_mentioned': allergen_result.allergens_mentioned,
                'allergen_awareness': allergen_result.allergen_awareness,
                'substitution_count': allergen_result.substitution_mentions,
                'has_safety_warnings': allergen_result.safety_notes_present,
            }
        }
