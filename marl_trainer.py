import os
import torch
from unsloth import FastLanguageModel
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

# 1. SETTINGS
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DATASET_FILE = "trl_dataset.jsonl"
OUTPUT_DIR = "nexus-student-7b"

def train():
    print(f"🚀 Starting DPO Fine-tuning: {MODEL_NAME}")
    
    # 2. LOAD MODEL & TOKENIZER (Optimized with Unsloth)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 4096,
        load_in_4bit = True,
    )

    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
    )

    # 3. LOAD DATASET
    if not os.path.exists(DATASET_FILE):
        print(f"❌ ERROR: {DATASET_FILE} not found. P2 must run reflexion.py first!")
        return

    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    # 4. CONFIGURE DPO
    training_args = DPOConfig(
        output_dir = OUTPUT_DIR,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        learning_rate = 5e-5,
        lr_scheduler_type = "cosine",
        num_train_epochs = 3,
        logging_steps = 1,
        save_steps = 50,
        optim = "adamw_8bit",
        bf16 = True,
        remove_unused_columns = False,
    )

    # 5. INITIALIZE TRAINER
    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None, # Unsloth handles reference model automatically to save VRAM
        args = training_args,
        train_dataset = dataset,
        tokenizer = tokenizer,
        beta = 0.1, # DPO temperature
        max_prompt_length = 1024,
        max_length = 2048,
    )

    # 6. TRAIN
    print("🔥 Training started...")
    dpo_trainer.train()

    # 7. SAVE
    print(f"✅ Training complete! Saving to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    train()