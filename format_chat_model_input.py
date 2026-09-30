import pandas as pd

INPUT_PATH = "data/advbench_base_model_completions.csv"

base_model_completions = pd.read_csv(INPUT_PATH)

def format_chat_input(goal: str, completion: str) -> str:
    return f"<s>[INST] {goal} [/INST] {completion} </s>"

