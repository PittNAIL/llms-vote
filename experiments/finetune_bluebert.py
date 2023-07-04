from datasets import load_dataset
import pandas as pd
import datasets

from datasets import Dataset
import numpy as np
import evaluate

from transformers import AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# dataset = load_dataset("yelp_review_full")
model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv("window_size_16.csv")
df_train, df_test = train_test_split(df, test_size=0.1, random_state=1234)
train_dataset = Dataset.from_dict(df_train)
test_dataset = Dataset.from_dict(df_test)
dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})


def tokenize_function(examples):
    return tokenizer(examples["window"], padding="max_length", truncation=True, max_length=128)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))

small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=5)


training_args = TrainingArguments(output_dir="test_trainer")

args = TrainingArguments(
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    weight_decay=0.01,
    warmup_steps=500,
    eval_accumulation_steps=1,
    output_dir="fine-tune-output",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    return {"precision": precision, "recall": recall, "f1": f1}


# def compute_metrics(eval_pred):
#    logits, labels = eval_pred
#    predictions = np.argmax(logits, axis=-1)
#    return {'accuracy': (predictions == labels).mean()}
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()


evaluation_result = trainer.evaluate()
print(evaluation_result)
