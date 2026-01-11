import os
# 1. Memory Allocation Config (Must be done before importing torch)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils.utils import get_device

def qlora_finetuning(model_name, dataset, output_dir="models/qwen-recipe-qlora"):
    torch.cuda.empty_cache()
    
    device_kwargs, use_qlora = get_device()

    # --- 2. Quantization Config ---
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if device_kwargs.get("torch_dtype") == torch.bfloat16 else torch.float16
        )

    # --- 3. Load Model ---
    clean_kwargs = {
        k: v for k, v in device_kwargs.items() 
        if k not in ["torch_dtype", "load_in_4bit", "load_in_8bit"]
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        quantization_config=bnb_config,
        **clean_kwargs
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "right" 

    # --- 4. Prepare for QLoRA ---
    if use_qlora:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

    model.config.use_cache = False 

    # Reduced rank and targets for VRAM efficiency
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 5. Formatting ---
    def formatting_func(example):
        messages = [
            {"role": "user", "content": example['instruction']},
            {"role": "assistant", "content": example['output']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        # Max length increased to 1024 to handle ingredients + steps
        return tokenizer(text, max_length=1024, truncation=True)

    print("Tokenizing and formatting dataset...")
    tokenized_dataset = dataset.map(
        formatting_func,
        batched=False,
        remove_columns=dataset["train"].column_names
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 6. Training Arguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1, 
        per_device_eval_batch_size=1, 
        gradient_accumulation_steps=16, 
        gradient_checkpointing=True,
        learning_rate=2e-4,
        logging_steps=10,
        eval_strategy="steps",      
        eval_steps=100,             
        save_strategy="steps",
        save_steps=100,             
        num_train_epochs=1,        
        bf16=(device_kwargs.get("torch_dtype") == torch.bfloat16), 
        fp16=(device_kwargs.get("torch_dtype") == torch.float16),
        optim="paged_adamw_8bit",
        report_to="none",
        group_by_length=True,
        max_grad_norm=0.3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    print("🚀 Training started...")
    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved in '{output_dir}'")
