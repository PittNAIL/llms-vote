import argparse
import numpy as np
import pandas as pd

import datasets
from datasets import Dataset

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

MODEL_NAME = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser("csv file for finetuning BlueBERT")

    parser.add_argument(
        "--data", type=str, help="path to dataset", required=True
    )  # E: line too long (86 > 79 characters)
    return parser.parse_args()


def tokenize_function(examples):
    return tokenizer(examples["window"], padding="max_length", truncation=True, max_length=128)


model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


training_args = TrainingArguments(
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


def main() -> None:
    """Finetunes BlueBERT from input dataset."""

    args = parse_args()
    df = pd.read_csv(args.data)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=1234)

    train_dataset = Dataset.from_dict(df_train)
    test_dataset = Dataset.from_dict(df_test)
    dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    evaluation_result = trainer.evaluate()
    print(evaluation_result)


if __name__ == "__main__":
    main()
