cd workspace
git clone https://github.com/alexroto/llm-safety-finetuning.git
cd llm-safety-finetuning
pip install -r requriements.txt
export HF_TOKEN=""
python lora.py
python post_finetune_refesual_evaluation.py
python harmful_content_evaluation.py
