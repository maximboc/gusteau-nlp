import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from src.utils.utils import get_device
from src.finetuning.qlora.utils.utils import format_example

def qlora_finetuning(model_name, dataset, output_dir="models/qwen-recipe-qlora"):
    # --- 1. Device management ---
    # N.B. QLora can be tested only on GPU
    # However Lora is also availabe on CPU and MPS chips
    device_kwargs, use_qlora = get_device()

    # --- 2. Load model ---
    model = AutoModelForCausalLM.from_pretrained(model_name, **device_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # --- 3. Prepare QLoRA / PEFT if supported ---
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. Tokenization function ---
    tokenized_dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        batched=True,    # tokenize multiple examples at once
        num_proc=os.cpu_count()  # parallelize across cores
    )

    # --- 5. TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=200,
        num_train_epochs=2,
        bf16=(device_kwargs["torch_dtype"]==torch.bfloat16),
        fp16=(device_kwargs["torch_dtype"]==torch.float16),
        optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to="none",
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
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved in '{output_dir}'")
