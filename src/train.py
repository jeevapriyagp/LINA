import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score

# Disable wandb in Colab
os.environ["WANDB_DISABLED"] = "true"

# ======================
# Load emotion labels
# ======================
with open("data/emotions.txt", "r") as f:
    emotions = [line.strip() for line in f.readlines()]

num_labels = len(emotions)

# ======================
# Dataset class
# ======================
class GoEmotionsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len=128):
        self.data = pd.read_csv(file_path, sep="\t", names=["text", "labels"])
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["text"])
        labels = str(self.data.iloc[idx]["labels"])

        label_ids = np.zeros(len(emotions))
        for l in labels.split(","):
            if l.isdigit():
                label_ids[int(l)] = 1

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label_ids, dtype=torch.float)
        }

# ======================
# Metrics
# ======================
def compute_metrics(eval_pred):
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.3).int().numpy()
    f1 = f1_score(labels, preds, average="micro", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def train_model():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )

    train_dataset = GoEmotionsDataset("data/train.tsv", tokenizer)
    dev_dataset = GoEmotionsDataset("data/dev.tsv", tokenizer)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_dir="./logs",
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save model
    save_dir = "models/emotion_model"
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Model and tokenizer saved at {save_dir}")

if __name__ == "__main__":
    train_model()
