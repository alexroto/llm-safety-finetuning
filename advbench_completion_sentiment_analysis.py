import os
import pandas as pd
from tqdm import tqdm

from transformers import pipeline
import torch

torch.manual_seed(123)

BATCH_SIZE = 8
HF_TOKEN = os.getenv("HF_TOKEN")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pre-trained sentiment analysis pipeline
classifier = pipeline("sentiment-analysis",
                      model="distilbert-base-uncased-finetuned-sst-2-english",
                      token=HF_TOKEN,
                      device=device)

advbench_completions = pd.read_csv("advbench_completions.csv")
completions = advbench_completions.completion.to_list()

labels = []
scores = []

def extract_labels_and_scores(batch_results):
    _labels = [x.get("label") for x in batch_results]
    _scores = [x.get("score") for x in batch_results]
    return _labels, _scores

for i in tqdm(range(0, len(completions), BATCH_SIZE)):
    batch = completions[i : i + BATCH_SIZE]
    results = classifier(batch)
    batch_labels, batch_scores = extract_labels_and_scores(batch_results=results)
    labels.extend(batch_labels)
    scores.extend(batch_scores)

advbench_completions["sentiment_analysis_label"] = labels
advbench_completions["sentiment_analysis_score"] = scores

label_summary = advbench_completions[['completion', 'sentiment_analysis_label']].groupby('sentiment_analysis_label').count()
print("=======Count of Labels=======")
print(label_summary)

advbench_completions.to_csv("advbench_completions_with_sentiment_analysis_labels_and_scores.csv")