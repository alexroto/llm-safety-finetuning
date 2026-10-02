"""
LoRA finetune Llama-2-7b-chat-hf on (goal -> compliant completion) pairs,
to study how finetuning shifts refusal behavior.

Run from a shell, inside tmux, so it survives disconnects:

    python lora.py
"""
import os

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

torch.manual_seed(123)
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
DATA_PATH = "data/advbench_completions_with_analysis_labels_and_scores.csv"  # your cleaned dataset
OUTPUT_DIR = "llama2-7b-chat-refusal-lora"

USE_4BIT = True  # set False if VRAM >=24GB and want full fp16 LoRA instead

MAX_LENGTH = 512
VAL_FRACTION = 0.15
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4  # effective batch size = PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS

# ---------------------------------------------------------------------------
# Load + filter data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
assert {"goal", "completion"}.issubset(df.columns)

# Drop anything flagged as a refusal, empty, or otherwise bad from the earlier
# analysis pass -- we only want genuinely compliant completions as training targets.
if "looks_like_refusal_flag" in df.columns:
    before = len(df)
    df = df[df["looks_like_refusal_flag"] == 0]
    print(f"Dropped {before - len(df)} refusal-flagged rows; {len(df)} remain.")

df = df[df["completion"].notna() & (df["completion"].str.strip() != "")].reset_index(drop=True)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # right-padding is correct for training (not generation)


def build_example(goal: str, completion: str) -> dict:
    """
    Tokenize prompt and full (prompt + completion) separately so we know exactly
    where the prompt ends, then mask prompt tokens out of the loss with -100.
    """
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": goal}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = prompt_text + " " + completion.strip() + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH
    )["input_ids"]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100  # mask prompt tokens out of the loss

    return {"input_ids": full_ids, "labels": labels}


examples = [build_example(g, c) for g, c in zip(df["goal"], df["completion"])]
dataset = Dataset.from_list(examples)
dataset = dataset.train_test_split(test_size=VAL_FRACTION, seed=123)
print(f"Train examples: {len(dataset['train'])} | Val examples: {len(dataset['test'])}")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
if USE_4BIT:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN, torch_dtype=torch.bfloat16, device_map="auto"
    )

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    bf16=True,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="none",  # set to "wandb" if you've got it configured
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
)

trainer.train()

# ---------------------------------------------------------------------------
# Save adapter only (small, portable -- not the full merged model)
# ---------------------------------------------------------------------------
model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
print(f"Saved LoRA adapter to {OUTPUT_DIR}/final_adapter")