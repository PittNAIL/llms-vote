import argparse
import datasets
import numpy as np
import pandas as pd
import torch

from torch import nn

from datasets import Dataset

from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer
from transformers import Trainer, TrainingArguments
from transformers import set_seed

set_seed(1234)

MODEL_NAME = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""

    parser = argparse.ArgumentParser("csv file for finetuning BlueBERT")

    parser.add_argument("--data", type=str, help="path to dataset", required=True)
    return parser.parse_args()


def tokenize_function(examples):
    return tokenizer(examples["window"], padding="max_length", truncation=True, max_length=128)


class mentionBERT(nn.Module):
    def __init__(self, bert_model_name):
        super(mentionBERT, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)

        hidden_states = outputs.hidden_states
        second_to_last_layer_hs = hidden_states[-2]

        averaged_representations = torch.mean(second_to_last_layer_hs, dim=1)
        pooled_output = self.dense(averaged_representations)
        pooled_output = self.activation(pooled_output)

        logits = self.classifier(pooled_output)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.bert.config.num_labels), labels.view(-1))

        return loss, logits


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
    output_dir="fine-tune-8",
)


def set_dir(data_file):
    """Get model path from args.data input"""
    if data_file == "window_size_4.csv":
        path = "fine-tune-4"
    elif data_file == "window_size_8.csv":
        path = "fine-tune-8"
    elif data_file == "window_size_16.csv":
        path = "fine-tune-16"
    elif data_file == "window_size_32.csv":
        path = "fine-tune-32"
    elif data_file == "window_size_64.csv":
        path = "fine-tune-32"
    else:
        raise ValueError("Unsupported data file for finetuning!")
    return path


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
    training_args.output_dir = set_dir(args.data)
    print(training_args.output_dir)
    train_dataset = Dataset.from_dict(df_train)
    test_dataset = Dataset.from_dict(df_test)
    dataset = datasets.DatasetDict({"train": train_dataset, "test": test_dataset})

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    small_train_dataset = tokenized_datasets["train"].shuffle(seed=1234).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=1234).select(range(1000))

    model = mentionBERT(MODEL_NAME)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    evaluation_result = trainer.evaluate()
    print(evaluation_result)


if __name__ == "__main__":
    main()
