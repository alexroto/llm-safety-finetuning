cd llm-safety-finetuning || exit
pip install -r requriements.txt
export HF_TOKEN=""
python lora.py
python post_finetune_refesual_evaluation.py
python harmful_content_evaluation.py
