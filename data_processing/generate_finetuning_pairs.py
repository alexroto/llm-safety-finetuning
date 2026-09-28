import os
import gc
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

# 1. Define the model repository ID from the Hugging Face Hub
model_id = "meta-llama/Llama-2-7b-hf"

# 2. Load the tokenizer and the model weights
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN).to(device)

files = [file for file in os.listdir('data') if file.endswith('.csv')]
print(f"files located = {files}")

harmful_behaviors = pd.read_csv('data/harmful_behaviors.csv')

def wrap_prompts(raw_prompts: list[str]) -> list[str]:
    few_shot_prefix = (
    "Question: What is the capital of France?\n"
    "Answer: The capital of France is Paris.\n\n"
    "Question: How many moons does Jupiter have?\n"
    "Answer: Jupiter has 95 known moons as of recent counts.\n\n"
    "Question: {prompt}\n"
    "Answer:"
    )
    return [few_shot_prefix.format(prompt=p) for p in raw_prompts]

def extract_answer(decoded_text: str) -> str:
    # Everything after the last "Answer:" is the model's response to our actual prompt
    answer = decoded_text.split("Answer:")[-1]
    # Cut off anything after the model starts a new Question/section
    for stop_marker in ["Question:", "###", "\nQ:"]:
        if stop_marker in answer:
            answer = answer.split(stop_marker)[0]
    return answer.strip()

raw_prompts = harmful_behaviors.goal.to_list()

CHECKPOINT_PATH = "advbench_completions.csv"
FAILED_LOG_PATH = "advbench_failed_prompts.csv"
BATCH_SIZE = 8  # start conservative on a 7B model; bump up if memory allows

# --- resume logic: skip prompts already completed in a prior run ---
if os.path.exists(CHECKPOINT_PATH):
    done_df = pd.read_csv(CHECKPOINT_PATH)
    already_done = set(done_df["goal"])
    results = done_df.to_dict("records")
    print(f"Resuming: {len(already_done)} prompts already completed.")
else:
    already_done = set()
    results = []

failed_prompts = []

remaining_prompts = [p for p in raw_prompts if p not in already_done]
wrapped_few_shot_prompts = wrap_prompts(raw_prompts=remaining_prompts)

for i in tqdm(range(0, len(wrapped_few_shot_prompts), BATCH_SIZE)):
    batch_prompts = wrapped_few_shot_prompts[i : i + BATCH_SIZE]
    batch_raw = remaining_prompts[i : i + BATCH_SIZE]

    tokenized_input = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        padding_side="left",
        max_length=100,
    )
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    try:
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_input,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.15,
                top_p=0.9,
            )
    except Exception as e:
        print(f"Batch {i} failed: {e}")
        print(f"Prompts in failed batch: {batch_raw}")
        failed_prompts.extend(batch_raw)
        pd.DataFrame({"goal": failed_prompts}).to_csv(FAILED_LOG_PATH, index=False)

        # free whatever we can and skip to the next batch
        del tokenized_input
        gc.collect()
        torch.cuda.empty_cache()
        continue

    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_output = [extract_answer(o.replace("\n", " ")) for o in decoded_output]

    for raw_prompt, completion in zip(batch_raw, decoded_output):
        results.append({"goal": raw_prompt, "completion": completion})

    # free GPU memory between batches
    del tokenized_input, outputs
    gc.collect()
    torch.cuda.empty_cache()

    # checkpoint to disk every batch so a crash doesn't lose progress
    pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)

print(f"Done. {len(results)} completions saved to {CHECKPOINT_PATH}")
if failed_prompts:
    print(f"{len(failed_prompts)} prompts failed and were logged to {FAILED_LOG_PATH}")