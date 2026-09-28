"""
Generate base-model completions for AdvBench prompts (few-shot framed).

Run from a shell so a poisoned CUDA context can be recovered by restarting:

    until python generate_completions.py; do echo "crashed, restarting"; sleep 5; done

To debug a CUDA assert with an accurate stack trace:

    DEBUG_CUDA=1 python generate_completions.py

Note: CUDA_LAUNCH_BLOCKING must be set before CUDA initializes, which is why it
is handled at the very top of this file. (TORCH_USE_CUDA_DSA is a compile-time
flag and has no effect on stock PyTorch wheels, so it is intentionally omitted.)
"""
import gc
import json
import os
import sys

DEBUG_CUDA = os.environ.get("DEBUG_CUDA") == "1"
if DEBUG_CUDA:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HF_TOKEN = os.getenv("HF_TOKEN")

MODEL_ID = "meta-llama/Llama-2-7b-hf"
DATA_PATH = "data/harmful_behaviors.csv"
CHECKPOINT_PATH = "advbench_completions.csv"
FAILED_LOG_PATH = "advbench_failed_prompts.csv"
MARKER_PATH = "in_progress.json"
BATCH_SIZE = 8  # start conservative on a 7B model; bump up if memory allows

# Completions containing these strings are contamination from the few-shot
# examples (e.g. from an earlier run where prompts were truncated) and get regenerated.
CONTAMINATION_MARKERS = ["95 known moons", "capital of France is Paris"]

# ---------------------------------------------------------------------------
# Model + tokenizer
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # required for correct batched causal generation

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=HF_TOKEN).to(device)
model.eval()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def extract_answer(generated_text: str) -> str:
    """Trim a *newly generated* continuation at the first sign of a new section.

    This is applied to only the newly generated tokens (not the prompt), so we
    no longer need to split on the last "Answer:" -- which previously could
    silently grab a few-shot example if the prompt got truncated.
    """
    answer = generated_text
    for stop_marker in ["Question:", "###", "\nQ:", "Q:"]:
        if stop_marker in answer:
            answer = answer.split(stop_marker)[0]
    return answer.strip()


def is_bad_completion(completion) -> bool:
    if not isinstance(completion, str) or len(completion.strip()) == 0:
        return True
    return any(marker in completion for marker in CONTAMINATION_MARKERS)


def save_failed(prompts: list[str]) -> None:
    pd.DataFrame({"goal": prompts}).to_csv(FAILED_LOG_PATH, index=False)


def is_fatal_cuda_error(exc: Exception) -> bool:
    """Errors after which the CUDA context is unusable and the process must restart."""
    msg = str(exc)
    return any(s in msg for s in ["device-side assert", "NVML_SUCCESS", "CUDA error"])


# ---------------------------------------------------------------------------
# Load data + resume state
# ---------------------------------------------------------------------------
harmful_behaviors = pd.read_csv(DATA_PATH)
raw_prompts = harmful_behaviors.goal.to_list()

# Prompts that failed in previous runs (never retried automatically)
if os.path.exists(FAILED_LOG_PATH):
    failed_prompts = pd.read_csv(FAILED_LOG_PATH)["goal"].tolist()
else:
    failed_prompts = []

# If the last run died mid-batch, the marker tells us which prompts were in flight.
# Those likely triggered the crash, so log them as failed and skip them.
if os.path.exists(MARKER_PATH):
    with open(MARKER_PATH) as f:
        crashed_prompts = json.load(f)
    print(f"Previous run crashed mid-batch; logging {len(crashed_prompts)} prompts as failed.")
    failed_prompts.extend(p for p in crashed_prompts if p not in failed_prompts)
    save_failed(failed_prompts)
    os.remove(MARKER_PATH)

# Previously completed prompts, minus any bad/contaminated completions
if os.path.exists(CHECKPOINT_PATH):
    done_df = pd.read_csv(CHECKPOINT_PATH)
    bad_mask = done_df["completion"].apply(is_bad_completion)
    if bad_mask.any():
        print(f"Dropping {bad_mask.sum()} bad/contaminated completions from checkpoint (will regenerate).")
    done_df = done_df[~bad_mask]
    results = done_df.to_dict("records")
    already_done = set(done_df["goal"])
    print(f"Resuming: {len(already_done)} prompts already completed.")
else:
    results = []
    already_done = set()

skip = already_done | set(failed_prompts)
remaining_prompts = [p for p in raw_prompts if p not in skip]
wrapped_few_shot_prompts = wrap_prompts(raw_prompts=remaining_prompts)
print(f"{len(remaining_prompts)} prompts remaining.")

# ---------------------------------------------------------------------------
# Generation loop
# ---------------------------------------------------------------------------
for i in tqdm(range(0, len(wrapped_few_shot_prompts), BATCH_SIZE)):
    batch_prompts = wrapped_few_shot_prompts[i : i + BATCH_SIZE]
    batch_raw = remaining_prompts[i : i + BATCH_SIZE]

    # No truncation: AdvBench goals are short, and silent right-truncation would
    # cut off the trailing "Answer:" cue and corrupt the output.
    tokenized_input = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
    )

    # CPU-side sanity check: token ids must fit in the embedding table,
    # otherwise the GPU raises an opaque device-side assert.
    max_id = tokenized_input["input_ids"].max().item()
    if max_id >= model.config.vocab_size:
        print(f"[batch {i}] token id {max_id} >= vocab size {model.config.vocab_size}; skipping batch.")
        failed_prompts.extend(batch_raw)
        save_failed(failed_prompts)
        continue

    input_len = tokenized_input["input_ids"].shape[1]
    tokenized_input = {k: v.to(device) for k, v in tokenized_input.items()}

    # Record what's in flight so a hard crash can be attributed on restart
    with open(MARKER_PATH, "w") as f:
        json.dump(batch_raw, f)
    print(f"[batch {i}] first prompt: {batch_raw[0][:80]!r} | padded length: {input_len}", flush=True)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **tokenized_input,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                repetition_penalty=1.15,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
    except Exception as e:
        print(f"[batch {i}] failed: {e}", flush=True)

        if is_fatal_cuda_error(e):
            # CUDA context is poisoned; any further CUDA call (even empty_cache)
            # will re-raise. Leave the marker in place and exit non-zero so the
            # shell restart loop relaunches us and the batch is logged as failed.
            sys.exit(1)

        # Ordinary (recoverable) exception: log the batch and move on
        print(f"Prompts in failed batch: {batch_raw}")
        failed_prompts.extend(batch_raw)
        save_failed(failed_prompts)
        os.remove(MARKER_PATH)

        del tokenized_input
        gc.collect()
        torch.cuda.empty_cache()
        continue

    # Decode only the newly generated tokens (everything after the prompt)
    new_tokens = outputs[:, input_len:]
    decoded_output = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    decoded_output = [extract_answer(o.replace("\n", " ")) for o in decoded_output]

    for raw_prompt, completion in zip(batch_raw, decoded_output):
        results.append({"goal": raw_prompt, "completion": completion})

    # Free GPU memory between batches
    del tokenized_input, outputs, new_tokens
    gc.collect()
    torch.cuda.empty_cache()

    # Checkpoint every batch, then clear the in-flight marker
    pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
    os.remove(MARKER_PATH)

print(f"Done. {len(results)} completions saved to {CHECKPOINT_PATH}")
if failed_prompts:
    print(f"{len(failed_prompts)} prompts failed and were logged to {FAILED_LOG_PATH}")