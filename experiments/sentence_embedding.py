import torch
import tqdm

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer


MODEL_NAME: str = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"


def main() -> None:
    """Generates sentence embbeddings."""

    # Load the BERT model and tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize the sentence and mention (or 16, 32)
    df = pd.read_csv("window_size_64.csv")

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

    # emb = np.array(emb)
    # lbl = np.array(lbl)

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
    print(f"{precision_macro=.2f}")
    print(f"{recall_macro=.2f}")
    print(f"{fscore_macro=.2f}")

    print("\nWeighted Metrics")
    print("================")
    print(f"{precision_weighted=.2f}")
    print(f"{recall_weighted=.2f}")
    print(f"{fscore_weighted=.2f}")


if __name__ == "__main__":
    main()
