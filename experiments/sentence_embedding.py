import argparse
import torch
import tqdm

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from transformers import AutoModel, AutoTokenizer


MODEL_NAME: str = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"


def parse_args() -> argparse.Namespace():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser("Generate Sentence Embeddings of Context Windows")
    parser.add_argument("--data", type=str, help="path to input data", required=True)

    return parser.parse_args()


def main() -> None:
    """Generates sentence embbeddings."""

    args = parse_args()

    # Load the BERT model and tokenizer
    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    df = pd.read_csv(args.data)
    #    df_1 = df[df["label"] == 1].head(100)
    #    df_0 = df[df["label"] == 0].head(100)
    #    df = pd.concat([df_1, df_0])

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

    np.save(f"embedding_{args.data}.npy", emb)
    np.save(f"labels_{args.data}.npy", np.array(lbl))

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
