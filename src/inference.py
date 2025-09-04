import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

# Load emotions
with open("data/emotions.txt", "r") as f:
    emotions = [line.strip() for line in f.readlines()]

# Load model
model_dir = "models/emotion_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_emotion(text, top_k=3):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]

    # Get top-k indices
    top_indices = probs.argsort()[-top_k:][::-1]  # sort descending
    top_emotions = [(emotions[i], float(probs[i]) * 100) for i in top_indices]

    return top_emotions

if __name__ == "__main__":
    print("Emotion Detection (type 'exit' to quit)\n")
    while True:
        text = input("Enter a sentence: ")
        if text.lower() == "exit":
            break
        top_emotions = predict_emotion(text, top_k=3)
        print("Top emotions:")
        for emo, score in top_emotions:
            print(f"  {emo}: {score:.2f}%")
        print()
