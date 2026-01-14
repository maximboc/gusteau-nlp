import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model, TaskType, prepare_model_for_kbit_training
from src.utils.utils import get_device
from src.finetuning.qlora.utils.utils import format_example

def prompt_tuning_finetuning(model_name, dataset, output_dir="models/qwen-recipe-prompt-tuning"):
    print(f"🎨 Starting Prompt Tuning for {model_name}")
    
    # --- 1. Device management ---
    device_kwargs, use_qlora = get_device()
    
    # --- 2. Load model ---
    # We can still use 4-bit loading for the base model to save memory, 
    # even if we are only training the prompts.
    model = AutoModelForCausalLM.from_pretrained(model_name, **device_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Prepare Model ---
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    
    # Enable gradient checkpointing before PEFT to save memory
    model.gradient_checkpointing_enable()

    # Prompt Tuning Config
    # We use 8 virtual tokens and initialize them with a relevant text for better convergence
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=8,
        prompt_tuning_init_text="Create a cooking recipe:",
        tokenizer_name_or_path=model_name,
    )
    
    model = get_peft_model(model, peft_config)
    
    # Enable gradient checkpointing to save memory if needed, though Prompt Tuning is light.
    # model.gradient_checkpointing_enable() 
    
    model.print_trainable_parameters()

    # --- 4. Tokenization function ---
    # Reduce max_length to 512 to save memory (recipes can still fit reasonably)
    tokenized_dataset = dataset.map(
        lambda x: format_example(x, tokenizer, max_length=512),
        batched=True,
        num_proc=os.cpu_count()
    )

    # --- 5. TrainingArguments ---
    # Prompt Tuning converges slower than LoRA, so we might need more epochs or higher LR.
    # A standard LR for prompt tuning is 0.3 (much higher than LoRA's 2e-4)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Reduced to 1 for memory constraints
        gradient_accumulation_steps=16, # Increased to maintain effective batch size
        learning_rate=0.3, # Higher learning rate for Prompt Tuning
        logging_steps=10,
        save_steps=200,
        num_train_epochs=3, # Reduced epochs for faster training
        bf16=(device_kwargs.get("torch_dtype")==torch.bfloat16),
        fp16=(device_kwargs.get("torch_dtype")==torch.float16),
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to="none",
        gradient_checkpointing=True, # Enable gradient checkpointing to save memory
        max_grad_norm=0.3,
    )

    # --- 6. Trainer ---
    data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    # --- 7. Train ---
    trainer.train()

    # --- 8. Save model and tokenizer ---
    os.makedirs(output_dir, exist_ok=True)
    trainer.save_model(output_dir) # Use trainer.save_model to save adapter properly
    tokenizer.save_pretrained(output_dir)
    print(f"Prompt Tuning adapter and tokenizer saved in '{output_dir}'")
