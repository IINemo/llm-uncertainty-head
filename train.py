import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch

# --- Configuration ---
model_name = "Qwen/Qwen3-0.6B"
dataset_path = {
    "train": "train_claims.jsonl",
    "validation": "dev_claims.jsonl"
}
output_dir = "./qwen3-06b-qlora-claims"

# --- Load dataset ---
dataset = load_dataset("json", data_files=dataset_path)

# --- Load tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Make sure there's a pad token

# --- Preprocessing with target-only loss ---
def preprocess(example):
    prompt = example["input_text"]
    target = example["target_text"]
    full_text = f"{prompt}\n{target}"

    enc = tokenizer(full_text, max_length=512, truncation=True, padding="max_length")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # Compute the start of the target_text portion
    prompt_enc = tokenizer(prompt, truncation=True, max_length=512)
    target_start = len(prompt_enc["input_ids"])

    # Mask the prompt part from contributing to the loss
    labels = [-100] * target_start + input_ids[target_start:]
    labels = labels[:512]  # ensure same length

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_datasets = dataset.map(preprocess, remove_columns=dataset["train"].column_names)

# --- Load model with QLoRA config ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# --- Training setup ---
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=20,
    learning_rate=2e-4,
    fp16=True,
    remove_unused_columns=False,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"]
)

# --- Train ---
trainer.train()

# --- Save final adapter and tokenizer ---
model.save_pretrained(output_dir + "/final")
tokenizer.save_pretrained(output_dir + "/final")
print("✅ Training complete. Model saved to:", output_dir + "/final")
