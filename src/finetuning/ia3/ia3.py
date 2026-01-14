"""
src/finetuning/ia3/ia3.py

IA³ fine-tuning implementation that integrates with the existing project structure.
GPU-optimized version with dtype consistency fixes.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, IA3Config, TaskType, PeftModel
from tqdm import tqdm
import os
import glob

def ia3_finetuning(
    model_name: str,
    dataset_dict,
    output_dir: str = "models/qwen-recipe-ia3",
    num_epochs: int = 3,
    batch_size: int = 4,  # Increased for GPU
    gradient_accumulation_steps: int = 4,  # Adjusted for GPU
    learning_rate: float = 5e-4,
    max_length: int = 512,
):
    """
    Fine-tune a model using IA³ (Infused Adapter by Inhibiting and Amplifying Inner Activations)
    
    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        dataset_dict: Dataset dictionary with 'train' and 'test' splits
        output_dir: Directory to save the fine-tuned adapter
        num_epochs: Number of training epochs
        batch_size: Batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate for training
        max_length: Maximum sequence length for tokenization
    """
    
    print(f"Setting up IA³ fine-tuning for {model_name}")
    print(f"Training samples: {len(dataset_dict['train'])}")
    print(f"Test samples: {len(dataset_dict['test'])}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🔥 Using device: {device.upper()}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add padding token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    print("Loading base model...")
    
    # CRITICAL: Use bfloat16 for GPU training to match inference dtype
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,  # Match inference dtype
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure IA³
    print("Configuring IA³ adapter...")
    
    # For Qwen models, target these modules
    ia3_config = IA3Config(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["k_proj", "v_proj", "down_proj"],  # Qwen architecture
        feedforward_modules=["down_proj"],
        inference_mode=False,
    )
    
    # Apply IA³ to model
    model = get_peft_model(model, ia3_config)
    
    # Print trainable parameters
    print("\nTrainable Parameters:")
    model.print_trainable_parameters()
    
    # Prepare dataset for training
    print("\nPreparing dataset...")
    
    def formatting_func(examples):
        """Format the dataset to match your existing structure"""
        texts = []
        for instruction, output in zip(examples['instruction'], examples['output']):
            # Match your existing prompt format
            text = f"{instruction}\n\n{output}"
            texts.append(text)
        return {"text": texts}
    
    # Format datasets
    train_dataset = dataset_dict['train'].map(
        formatting_func,
        batched=True,
        remove_columns=dataset_dict['train'].column_names
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    print("Tokenizing training data...")
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Set up training arguments (GPU-optimized)
    print("\nSetting up training arguments...")
    
    # Determine if bf16 is available
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    
    print(f"Mixed precision: bf16={use_bf16}, fp16={use_fp16}")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        bf16=use_bf16,  # Use bfloat16 if available
        fp16=use_fp16,  # Fallback to fp16
        dataloader_num_workers=2 if device == "cuda" else 0,  # More workers for GPU
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=False,  # Usually not needed for IA³
        optim="adamw_torch",
        max_grad_norm=1.0,  # Gradient clipping
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("\nStarting IA³ training...")
    print(f"Estimated time: Much faster than LoRA/QLoRA!")
    print(f"Model will be saved to: {output_dir}")
    print("\nYou can stop training with Ctrl+C and resume later.\n")
    
    def get_last_checkpoint(output_dir):
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
        if len(checkpoints):
            return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
        return None

    last_checkpoint = get_last_checkpoint(output_dir)

    trainer.train(resume_from_checkpoint=last_checkpoint) 

    # Save final model
    print("\nTraining complete! Saving model...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nIA³ adapter saved to: {output_dir}")
    print("Training metrics saved in the output directory.")
    
    return model, tokenizer


def test_ia3_generation(model_name, adapter_path, test_prompt):
    """
    Quick test to verify the IA³ model works
    
    Args:
        model_name: Base model name
        adapter_path: Path to IA³ adapter
        test_prompt: Test instruction to generate recipe
    """
    print(f"\nTesting IA³ model generation...")
    
    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model with same dtype as training
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,  # CRITICAL: Match training dtype
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load IA³ adapter
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # Generate
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n" + "="*60)
    print("GENERATED RECIPE:")
    print("="*60)
    print(generated_text)
    print("="*60)
    
    return generated_text