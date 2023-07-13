import argparse
import os
import shutil
import torch
import tqdm

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer


MODEL_NAME: str = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
experiments_dir = os.path.abspath(os.path.dirname(__file__))
CONFIG_FILE_PATH = os.path.join(experiments_dir, "config.json")


def parse_args() -> argparse.Namespace():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser("Generate Sentence Embeddings of Context Windows")
    parser.add_argument("--data", type=str, help="path to input data", required=True)
    parser.add_argument("--finetuned", action="store_true", default=False, help="Flag for using a finetuned model")

    return parser.parse_args()


def main() -> None:
    """Generates sentence embbeddings."""
    args = parse_args()

    data_file = args.data
    finetune_flag = args.finetuned
    if finetune_flag:
        if data_file == "window_size_8.csv":
            model_path = "fine-tune-8/checkpoint-189"
        elif data_file == "window_size_16.csv":
            model_path = "fine-tune-16/checkpoint-189"
        elif data_file == "window_size_32.csv":
            model_path = "fine-tune-32/checkpoint-189"
        else:
            raise ValueError("Unsupported data file for finetuning!")
        model_dir = os.path.join(experiments_dir, model_path)
        shutil.copy(CONFIG_FILE_PATH, model_dir)
        model = AutoModel.from_pretrained(model_dir)
    else:
        model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    df = pd.read_csv(data_file)

    emb = []
    lbl = []
    with torch.no_grad():
        for _, row in tqdm.tqdm(df.iterrows(), desc="Encoding dataset"):
            # This will encode the entire "sentence", being the full context window of 16,32,64
            # words, and return its vector representation.
            inputs = tokenizer(row["window"], return_tensors="pt")

            outputs = model(**inputs)
            # NOTE: sentence embedding is averaged in line with what Dr. Dong does.
            sentence_embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)

            emb.append(sentence_embedding)
            lbl.append(row["label"])

    emb = np.array([tensor.numpy() for tensor in emb])

    emb_train, emb_test, lbl_train, lbl_test = train_test_split(
        emb, lbl, test_size=0.1, random_state=1337
    )

    clf = LogisticRegression(max_iter=1_000)
    clf.fit(emb_train, lbl_train)
    pred = clf.predict(emb_test)

    precision_macro, recall_macro, fscore_macro, _ = precision_recall_fscore_support(
        pred, lbl_test, average="macro"
    )
    precision_weighted, recall_weighted, fscore_weighted, _ = precision_recall_fscore_support(
        pred, lbl_test, average="weighted"
    )

    print("Macro Metrics")
    print("================")
    print(f"precision_macro={precision_macro:.2f}")
    print(f"recall_macro={recall_macro:.2f}")
    print(f"fscore_macro={fscore_macro:.2f}")

    print("\nWeighted Metrics")
    print("================")
    print(f"precision_weighted={precision_weighted:.2f}")
    print(f"recall_weighted={recall_weighted:.2f}")
    print(f"fscore_weighted={fscore_weighted:.2f}")


if __name__ == "__main__":
    main()
