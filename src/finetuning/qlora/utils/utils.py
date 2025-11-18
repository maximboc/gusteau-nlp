def format_example(batch, tokenizer, max_length=1024):
    # 1. Create a list of prompts, one for each example in the batch
    prompts = [
        f"Instruction: {instruction}\n\nRecipe:" 
        for instruction in batch["instruction"]
    ]
    
    # 2. Create a list of full texts
    # (This assumes 'output' is a string, as implied by your conversion script)
    full_texts = [
        prompt + " " + output 
        for prompt, output in zip(prompts, batch["output"])
    ]
    
    # 3. Tokenize the entire list of texts at once (this is fast)
    tokens = tokenizer(
        full_texts, max_length=max_length, truncation=True, padding="max_length"
    )
    
    # 4. Create labels for the batch
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
