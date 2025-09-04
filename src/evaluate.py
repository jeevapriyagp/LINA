import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from train import GoEmotionsDataset, compute_metrics
import os

# Load model
model_dir = "models/emotion_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)

test_dataset = GoEmotionsDataset("data/test.tsv", tokenizer)

training_args = {
    "output_dir": "./results"
}

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("📊 Evaluation Results:")
print(trainer.evaluate(test_dataset))
