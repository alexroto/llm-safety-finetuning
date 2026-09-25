from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. Define the model repository ID from the Hugging Face Hub
model_id = "meta-llama/Llama-2-7b-hf"

# 2. Load the tokenizer and the model weights
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 3. Prepare an input string
text = "Hello, I am a language model,"
inputs = tokenizer(text, return_tensors="pt")

# 4. Generate a response
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
