import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
try:
    import outlines
    OUTLINES_AVAILABLE = True
except ImportError as e:
    OUTLINES_AVAILABLE = False
    print(f"Warning: Outlines not found ({e}). Constrained generation will be skipped in showcase.")

def generate_text(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            do_sample=True, 
            temperature=0.7
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_qualitative_showcase(dataset, base_model_id, adapter_path):
    print("\n" + "="*80)
    print("✨ QUALITATIVE SHOWCASE: Original vs Base vs Fine-Tuned vs DSPy vs Outlines ✨")
    print("="*80 + "\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Base Model: {base_model_id} on {device}...")
    
    # 1. Load Base Model
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id).to(device)

    # We select 2 samples instead of 3
    samples = dataset.select(range(min(2, len(dataset))))

    results = []

    print("Generating with Base Model...")
    for item in samples:
        # item['instruction'] already contains "Create a detailed recipe for {name}." 
        # So we shouldn't add "Create a recipe for" again.
        
        # 1. Base Model uses the raw instruction
        prompt_text = item['instruction'] 
        
        # NOTE: For Qwen-Instruct, ideally we would use the chat template:
        # prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": item['instruction']}], tokenize=False, add_generation_prompt=True)
        # But to keep it comparable to the fine-tuning (which didn't use chat templates), we'll try to keep it simple.
        # However, to give the base model a fair chance, let's just use the instruction directly or a simple wrapper.
        
        base_output = generate_text(base_model, tokenizer, prompt_text)
        results.append({
            "instruction": item['instruction'],
            "ground_truth": item['output'],
            "base_output": base_output
        })

    # 2. Load Adapter (Prompt Tuning)
    print(f"Loading Adapter from {adapter_path}...")
    try:
        model_tuned = PeftModel.from_pretrained(base_model, adapter_path)
    except Exception as e:
        print(f"Error loading adapter: {e}")
        return

    print("Generating with Fine-Tuned Model...")
    for i, item in enumerate(samples):
        # CORRECT PROMPT FORMAT MATCHING TRAINING: "Instruction: {content}\n\nRecipe:"
        prompt_text = f"Instruction: {item['instruction']}\n\nRecipe:"
        tuned_output = generate_text(model_tuned, tokenizer, prompt_text)
        results[i]["tuned_output"] = tuned_output

    # 3. DSPy (Prompt Engineered)
    print("Generating with DSPy Prompt...")
    for i, item in enumerate(samples):
        # Prompt from dspy_optimizer.py
        dspy_prompt = f"""You are a professional chef. Generate a detailed recipe for the following dish.

Dish: {item['instruction']}

Recipe (with Ingredients and Instructions):"""
        
        dspy_output = generate_text(model_tuned, tokenizer, dspy_prompt)
        results[i]["dspy_output"] = dspy_output

    # 4. Outlines
    print("Generating with Outlines (Constrained)...")
    if OUTLINES_AVAILABLE:
        try:
            # We pass the model directly
            # Note: outlines 0.1+ uses outlines.models.Transformers
            outlines_model = outlines.models.Transformers(model_tuned, tokenizer)
            
            # Regex for Ingredients and Instructions (Stricter)
            # Match "Ingredients:", then content, then explicit "Instructions:", then content.
            recipe_regex = r"Ingredients:[\s\S]+?\n\nInstructions:[\s\S]+"
            
            # Create the constraint type
            constraint = outlines.regex(recipe_regex)
            
            # Create the generator using the new API (Generator class)
            generator = outlines.Generator(outlines_model, constraint)
            
            for i, item in enumerate(samples):
                # CORRECT PROMPT FORMAT MATCHING TRAINING
                prompt_text = f"Instruction: {item['instruction']}\n\nRecipe:"
                
                print(f"  > Generating sample {i+1} with Outlines...")
                try:
                    # Outlines generation
                    # As of recent versions, samplers are deprecated.
                    # We pass generation arguments directly to the generator
                    # 'max_new_tokens' is the standard HF argument, 'max_tokens' might be deprecated or specific to other backends.
                    constrained_output = generator(prompt_text, max_new_tokens=256, temperature=0.7)
                    results[i]["outlines_output"] = constrained_output
                except Exception as inner_e:
                    print(f"    ! Sample {i+1} failed: {inner_e}")
                    results[i]["outlines_output"] = f"ERROR: {inner_e}"
                
        except Exception as e:
            print(f"❌ Outlines initialization failed: {e}")
            import traceback
            traceback.print_exc()
            for i, _ in enumerate(samples):
                results[i]["outlines_output"] = f"FAILED STARTUP: {e}"
    else:
        for i, _ in enumerate(samples):
            results[i]["outlines_output"] = "SKIPPED (Outlines not installed)"

    # 4. Display & Log
    log_content = []
    
    header = "\n" + "="*80 + "\n✨ QUALITATIVE SHOWCASE LOG ✨\n" + "="*80 + "\n"
    log_content.append(header)
    print(header)

    for i, res in enumerate(results):
        block = []
        block.append("\n" + "#"*80)
        block.append(f"🍲 SAMPLE {i+1}: {res['instruction']}")
        block.append("#"*80)
        
        block.append("\n--- 1. ORIGINAL DATASET (Ground Truth) ---")
        block.append(res['ground_truth'])
        
        block.append("\n--- 2. ORIGINAL MODEL (Base) ---")
        out_base = res['base_output'].replace(res['instruction'], "").strip()
        block.append(out_base)
        
        block.append("\n--- 3. FINE-TUNED MODEL (Prompt Tuning) ---")
        out_tuned = res['tuned_output'].replace(f"Instruction: {res['instruction']}\n\nRecipe:", "").strip()
        block.append(out_tuned)
        
        block.append("\n--- 4. DSPy (Prompt Optimized) ---")
        dspy_prompt_clean = f"""You are a professional chef. Generate a detailed recipe for the following dish.

Dish: {res['instruction']}

Recipe (with Ingredients and Instructions):"""
        # Clean up the output by removing the prompt part
        # Note: generate_text returns full sequence. The prompt might be slightly different after tokenization/detokenization
        # so simple replace might fail if spacing is weird. But let's try.
        if "dspy_output" in res:
             out_dspy = res['dspy_output']
             # Try simple replace first
             out_dspy = out_dspy.replace(dspy_prompt_clean, "").strip()
             # Fallback if replace didn't work (e.g. strict string matching failed)
             if len(out_dspy) > len(dspy_prompt_clean) and out_dspy.startswith("You are a professional chef"):
                 # Heuristic crop
                 # Find the last part of prompt
                 split_marker = "Recipe (with Ingredients and Instructions):"
                 if split_marker in out_dspy:
                     out_dspy = out_dspy.split(split_marker)[-1].strip()
             block.append(out_dspy)
        else:
             block.append("N/A")

        block.append("\n--- 5. OUTLINES (Constrained) ---")
        out_outlines = res['outlines_output'] 
        block.append(out_outlines)
        block.append("\n")
        
        # Print to console
        for line in block:
            print(line)
            
        # Add to log content
        log_content.extend(block)

    # Save to file
    with open("qualitative_showcase.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))
    print(f"\n✅ Qualitative showcase saved to 'qualitative_showcase.log'")

