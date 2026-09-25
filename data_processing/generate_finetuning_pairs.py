import os
import re

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

HF_TOKEN = os.getenv("HF_TOKEN")

# 1. Define the model repository ID from the Hugging Face Hub
model_id = "meta-llama/Llama-2-7b-hf"

# 2. Load the tokenizer and the model weights
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_id, token=HF_TOKEN).to(device)

files = [file for file in os.listdir('data') if file.endswith('.csv')]
print(f"files located = {files}")

harmful_strings = pd.read_csv('data/harmful_strings.csv')
harmful_strings.head(2)

harmful_strings_targets = harmful_strings.target.to_list()
harmful_strings_input = tokenizer(harmful_strings_targets, padding=True, truncation=True, return_tensors="pt")
model_input = harmful_strings_input["input_ids"].to(device)

print(model_input.shape)
output = model.generate(model_input, max_new_tokens=20)
torch.save(output, 'harmful_strings_encoded_ouptut.pt')
decoded_ouptut = tokenizer.decode(output, skip_special_tokens=True)

cleaned_decoded_output = []

for target, output in zip(harmful_strings_targets, decoded_ouptut):
    cleaned_output = output.lstrip(target)
    cleaned_output = cleaned_output.replace("\n", ". ")
    cleaned_output = re.sub(r'[^A-Za-z0-9 ]', '', cleaned_output)
    cleaned_output = cleaned_output.strip()
    cleaned_output_tokens = cleaned_output.split(' ')
    cleaned_output_tokens = [token for token in cleaned_output_tokens if token]
    cleaned_output = ' '.join(cleaned_output_tokens)
    cleaned_decoded_output.append(cleaned_output)

harmful_strings['responses_before_finetuning'] = cleaned_decoded_output
harmful_strings.to_csv('harmful_strings_with_output_before_finetuning.csv', index=False)
